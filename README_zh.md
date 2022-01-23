# <div align=center style="line-height:1.5;font-size:40;"><b>Accelerated-TD</b></div>

我对加速的梯度时序差分算法（Accelerated Gradient Temporal Difference Learning algorithm, ATD）的Python实现。

<img src="translations.svg" style="height:25px" />

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/VEXLife/Accelerated-TD) ![GitHub](https://img.shields.io/github/license/VEXLife/Accelerated-TD) ![GitHub issues](https://img.shields.io/github/issues/VEXLife/Accelerated-TD)  ![Gitee issues](https://img.shields.io/badge/dynamic/json?label=Gitee%20Issues&query=%24.open_issues_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=flat) ![GitHub Pull requests](https://img.shields.io/github/issues-pr/VEXLife/Accelerated-TD) ![Contribution](https://img.shields.io/static/v1?label=contribution&message=welcome&color=%23ff66ef&link=http://github.com/VEXLife/Accelerated-TD) ![GitHub Repo stars](https://img.shields.io/github/stars/VEXLife/Accelerated-TD?style=social) ![GitHub forks](https://img.shields.io/github/forks/VEXLife/Accelerated-TD?style=social) ![Gitee Repo stars](https://img.shields.io/badge/dynamic/json?label=Gitee%20Stars&query=%24.stargazers_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=social) ![Gitee forks](https://img.shields.io/badge/dynamic/json?label=Gitee%20Forks&query=%24.forks_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=social)

# 介绍

## 智能体

`PlainATDAgent` 直接更新 ![](https://latex.codecogs.com/svg.image?\mathbf{A}) 矩阵而 `SVDATDAgent` 和 `DiagonalizedSVDATDAgent` 分别更新其奇异值分解。这在论文的作者（论文链接见后）看来具有更小的复杂度。 `SVDATDAgent` 和 `DiagonalizedSVDATDAgent` 的区别在于 `SVDATDAgent` 采用了这里提到的方法：[Brand 2006](https://pdf.sciencedirectassets.com/271586/1-s2.0-S0024379506X04573/1-s2.0-S0024379505003812/main.pdf) 而 `DiagonalizedSVDATDAgent` 采用了这里提到的方法[Gahring 2015](https://arxiv.org/pdf/1511.08495) 来对角化 ![](https://latex.codecogs.com/svg.image?\mathbf{\Sigma}) 矩阵以便矩阵的伪逆更容易计算。尽管这种方法我还没完全搞懂。
我还实现了一个名为 `TDAgent` 的传统梯度时间差异代理。我在下面介绍的几种环境中对它们进行了测试。

## 后端支持

我为 PyTorch(CPU) 提供了后端支持，以跳过从 `numpy.ndarray` 到 `torch.Tensor` 的转换过程，反之亦然。要想使用此支持，您可以在导入 `atd` 模块之前添加此代码：
```python
import os
os.environ["ATD_BACKEND"] = "NumPy"  # 或 "PyTorch"
```

如果您想要自己运行测试，只需克隆此仓库并运行 `python algorithm_test/<random_walk 或 boyans_chain>.py` 。:)

# 要求

- Python>=3.9
- NumPy>=1.19
- 如果你想使用 PyTorch 作为后端，那么还需要Torch>=1.10
- 如果你想运行我的测试脚本，那么还需要Matplotlib>=3.3.3
- 如果你想运行我的测试脚本，那么还需要Tqdm

# 测试

## 随机游走（Random Walk）

这个环境来自[Sutton的书](http://incompleteideas.net/book/RLbook2020.pdf)。

代码文件是[这个](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/random_walk.py)，结果[在这](https://github.com/VEXLife/加速-TD/blob/main/figures/random_walk.png)：
![random_walk](https://user-images.githubusercontent.com/36587232/144394107-d0cf9bd0-ea09-4e48-ba04-cb07af9e4240.png)

## Boyan的链（Boyan's Chain）

该环境是在[Boyan 1999](https://www.researchgate.net/publication/2621189_Least-Squares_Temporal_Difference_Learning)中提出的。

代码文件是[这个](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/boyans_chain.py)，结果[在这](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/boyans_chain.png)：
![boyans_chain](https://user-images.githubusercontent.com/36587232/144394100-dc8c48c2-1d38-433f-aea6-f202da3bbb13.png)

# 使用方法

要将我的算法实现导入您的项目，如果您对此不太熟悉，请按照这些说明进行操作。
1. 克隆存储库并将 `atd.py` 复制到您想要的位置。如果您从 GitHub 下载了 .zip 文件，请记得将其解压缩。
2. 将此代码添加到 Python 脚本的头部：
   ```python
   from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent # 或者你想要的任何智能体
   ```
3. 如果目标目录与您要执行的 Python 主程序文件所在的目录不同，您应该使用此代码片段而不是步骤 2 中的来将相应的目录附加到环境变量，以便 Python 解释器可以找到它。或者，您可以考虑较新版本 Python 提供的 `importlib` 。
   ```python
   import sys

   sys.path.append("<你放置 atd.py 的目录>")
   from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent # 或者你想要的任何智能体
   ```
4. 像这样初始化一个智能体，你就可以用它了！
   ```python
   agent = TDAgent(lr=0.01, lambd=0, observation_space_n=4, action_space_n=2)
   ```

参考文献：[Gahring 2016](https://arxiv.org/pdf/1611.09328.pdf)

随便提Pull Requests！我期待着您的评论。
