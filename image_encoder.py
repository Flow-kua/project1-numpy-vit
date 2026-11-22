"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # TODO-02：B 为图片数，N 为 token 数，D 为特征数。
        # 将每个token的768个特征拆为12个头，每头64个特征。
        B, N, D = x.shape

        x = x.reshape(B, N, self.heads, self.head_dim)

        # 各头内部的token顺序保持不变。
        x = x.transpose(0, 2, 1, 3)

        return x

    def __call__(self, x, causal=False):
        # 权重按[输出维度,输入维度]保存，矩阵乘法时需转置。
        # 一次生成 Q、K、V：[B,N,D] -> [B,N,3D]，使用已有参数，不进行训练。
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        # 沿特征轴依次取出 Q、K、V，再分别拆成 [B,H,N,D/H]。
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a：仅交换 K 的最后两个轴，让每个 Q 与所有 K 计算点积。
        # 得到 [B,H,N,N] 分数，除以sqrt调整尺度；保持 float32。
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(
            np.float32(self.head_dim)
        )

        # 因果掩码用于后续文本分支；图片使用 causal=False，可参考所有 token。
        if causal:
            scores = causal_mask(scores)
        # 已提供的 softmax 函数沿最后一轴计算权重，每行权重之和为 1。
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b：每个 token 按自己的权重汇总所有 V，得到 [B,H,N,D/H]。
        context = coefficients @ v
        # 先恢复 [B,N,H,D/H] 的轴顺序，再将各头特征合并为 [B,N,D]。
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape)

        # TODO-03c：输出投影重新组合各头的信息，返回形状仍为 [B,N,D]。
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # 先对每个 token 的特征做 LayerNorm，再计算注意力更新。
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a：第一次残差连接，将注意力更新加回原始 x，而非归一化后的输入。
        residual = x + update

        # 第二次 LayerNorm 使用独立参数，为 MLP 准备输入。
        u = self.norm2(residual)

        # TODO-04b：MLP 对每个 token 分别使用相同参数处理特征。
        # 第一层将 D 扩展为 4D，并应用 GELU：[B,N,768] -> [B,N,3072]。
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        # 第二层将特征还原为 D，便于与残差相加：[B,N,3072] -> [B,N,768]。
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        # 第二次残差连接加回 residual，模块输入、输出均为 [B,N,D]。
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01：仅重新组织像素，保持小块从左到右、从上到下的顺序。

        B = images.shape[0]
        # 六个轴依次为：图片、通道、小块行、块内行、小块列、块内列。
        patches = images.reshape(B, 3, 7, 32, 7, 32)
        # 调整为：图片、小块行、小块列、通道、块内行、块内列。
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        # 7*7=49 个小块；每块按 C,H,W 展开为 3*32*32=3072 个数。
        return patches.reshape(B, 49, 3072)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a：用 [3072,768] 权重把小块像素映射为特征，无偏置。
        # [B,49,3072] -> [B,49,768]。

        tokens = patches @ self.patch_weight.T

        # 为每张图片复制一行初始 CLS：[1,1,768] -> [B,1,768]。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b：沿 token 轴在最前面拼接 CLS，49 行变为 50 行。

        tokens = np.concatenate([cls, tokens], axis=1)

        # 位置特征 [50,768] 广播到每张图片，输出 [B,50,768]。
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # TODO-06：CLS 经各模块汇总图片信息，取第 0 行得到 [B,768]。

        cls = tokens[:, 0, :]
        # 对最终 CLS 的特征做独立的后置 LayerNorm。
        cls = self.post_norm(cls)
        # 最终投影权重已按 [输入维度,输出维度] 保存，不转置；输出 [B,512]。
        output = cls @ self.projection

        # 分别将每张图片的输出向量归一化为单位长度。
        return unit_vector(output)
