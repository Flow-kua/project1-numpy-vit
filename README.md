# Project 1：基于 NumPy 的 ViT 图像编码器

## 1. 项目简介

本项目使用 NumPy 实现 OpenCLIP ViT-B-32 图像编码器的前向计算。程序读取课程提供的模型参数，将图片转换为 512 维单位长度特征向量，不涉及模型训练，不需要 GPU、PyTorch 或 OpenCLIP 软件包。

处理流程：图片预处理 → 32×32 小块划分 → 小块投影、CLS 拼接与位置特征相加 → 前置 LayerNorm → 12 个 Transformer 模块 → CLS 提取与后置 LayerNorm → 512 维投影与单位长度归一化。

## 2. 文件与数据准备

本次补全的代码位于 `image_encoder.py`，六组 TODO 已实现：图片切块、多头拆分、注意力计算、Transformer 模块、小块嵌入和最终特征输出。

复现时，请先解压课程提供的 `project1-numpy-student` 附件，再将提交的 `image_encoder.py` 放到同名文件的位置。复现目录需包含以下配套内容：

| 文件或目录 | 用途 |
| --- | --- |
| `image_encoder.py` | 本次完成的图像编码器实现 |
| `helpers.py` | 课程提供的 LayerNorm、GELU、注意力权重和向量归一化工具 |
| `preprocess_image.py` | 图片尺寸调整、中心裁剪和像素归一化 |
| `runtime.py` | 加载模型参数和图片 |
| `run.py` | 运行编码器并保存特征 |
| `check.py` | 分阶段比较实现结果和参考结果 |
| `config.json`、`requirements.txt` | 模型配置和依赖清单 |
| `weights/` | 模型参数 `image_encoder.npz` 及校验清单 `manifest.json` |
| `assets/` | 图片、图片信息、参考结果及其校验清单 |

大型参数文件与参考数据由课程附件提供，不随提交材料重复上传。请保持课程附件的目录结构和参数配置。

## 3. 运行环境与依赖

- Python 3.9 或更新版本；依赖版本需与所用 Python 兼容。
- 依赖范围与 `requirements.txt` 一致：`numpy>=1.24,<3`、`Pillow>=9.2,<13`、`imageio>=2.31,<3`。其中 imageio 用于课堂示例，编码器运行与检查主要使用 NumPy 和 Pillow。

```bash
conda install "numpy>=1.24,<3" "pillow>=9.2,<13" "imageio>=2.31,<3"
```

当 conda 无法满足需求时，在选定的虚拟环境中使用：

```bash
python -m pip install -r requirements.txt
```

以下命令均在上述项目目录中执行。依赖和课程数据准备好后，计算过程不需要联网。

## 4. 检查与运行

### 4.1 检查全部阶段

```bash
python check.py --stage all
```

全部通过后，终端末尾显示：

```text
Requested numerical checks passed. The reference file is never an encoder input.
```

需要定位某个阶段时，也可以单独检查，例如：

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage output
```

检查采用形状、float32 数据类型、有限数值以及数值误差容差等条件。参考数组只用于检查，不参与编码器的前向计算。

### 4.2 处理课程提供的十张图片

```bash
python run.py --output output_tokens.npz
```

正常完成时显示：

```text
Saved (10, 512) to output_tokens.npz
```

输出文件保存在运行命令时的当前目录。

### 4.3 处理其他图片

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

将示例图片路径替换为实际路径。输入 B 张图片时，输出特征形状为 `[B,512]`。

## 5. 输出文件

`.npz` 是 NumPy 用于保存多个数组的文件格式。输出文件包含：

| 数组名称 | 内容 | 默认十张图片时的形状 |
| --- | --- | --- |
| `ids` | 图片 ID，与特征行一一对应 | `(10,)` |
| `output_tokens` | 图片的 float32 特征，每行长度归一化为 1 | `(10,512)` |

可在项目目录下的 Python 解释器中读取：

```python
import numpy as np

with np.load("output_tokens.npz", allow_pickle=False) as data:
    print(data["ids"])
    print(data["output_tokens"].shape)
    print(data["output_tokens"][0])
```

这些数值是图片特征，不是直接输出的分类名称。

## 6. 本地检查结果

以下记录来自本次在本地环境中执行 `python check.py --stage all` 的终端输出。不同软硬件环境下，浮点误差可能略有变化，应以检查程序的容差判定为准。

资源检查通过：152 个匹配的视觉参数张量，以及十组图片和参考结果。

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 | 结果 |
| --- | --- | --- | --- | --- |
| preprocess | `(10,3,224,224)` | `0.000e+00` | `0.000e+00` | PASS |
| patches | `(2,49,3072)` | `0.000e+00` | `0.000e+00` | PASS |
| tokens | `(2,50,768)` | `4.387e-05` | `1.335e-06` | PASS |
| block0 | `(2,50,768)` | `7.147e-06` | `3.348e-07` | PASS |
| last_tokens | `(2,50,768)` | `4.196e-05` | `2.633e-06` | PASS |
| output | `(10,512)` | `5.662e-07` | `8.688e-08` | PASS |
| batch | `(3,512)` | `4.061e-07` | `7.848e-08` | PASS |

中间部分阶段使用两张图片检查；最终输出阶段使用全部十张图片。batch 阶段检查调整顺序和重复图片时的结果。

数值匹配条件为 `abs(actual-reference) <= atol + rtol*abs(reference)`。预处理使用 `atol=2e-6`，最终输出和批量检查使用 `atol=2e-5`，其余数值阶段使用 `atol=2e-4`；这些阶段的 `rtol` 均为 `2e-4`。

## 7. 实现说明

模型参数、数据、预处理、数学辅助函数和运行检查脚本由课程提供。本次在课程框架中补全了 `image_encoder.py` 的六组 TODO，并完成本地检查与输出生成。
