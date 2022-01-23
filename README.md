# <div align=center style="line-height:1.5;font-size:40;"><b>Accelerated-TD</b></div>

My implementation of the Accelerated Gradient Temporal Difference Learning algorithm (ATD) in Python.

<img src="translations.svg" style="height:25px" />

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/VEXLife/Accelerated-TD) ![GitHub](https://img.shields.io/github/license/VEXLife/Accelerated-TD) ![GitHub issues](https://img.shields.io/github/issues/VEXLife/Accelerated-TD)  ![Gitee issues](https://img.shields.io/badge/dynamic/json?label=Gitee%20Issues&query=%24.open_issues_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=flat) ![GitHub Pull requests](https://img.shields.io/github/issues-pr/VEXLife/Accelerated-TD) ![Contribution](https://img.shields.io/static/v1?label=contribution&message=welcome&color=%23ff66ef&link=http://github.com/VEXLife/Accelerated-TD) ![GitHub Repo stars](https://img.shields.io/github/stars/VEXLife/Accelerated-TD?style=social) ![GitHub forks](https://img.shields.io/github/forks/VEXLife/Accelerated-TD?style=social) ![Gitee Repo stars](https://img.shields.io/badge/dynamic/json?label=Gitee%20Stars&query=%24.stargazers_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=social) ![Gitee forks](https://img.shields.io/badge/dynamic/json?label=Gitee%20Forks&query=%24.forks_count&url=http%3A%2F%2Fgitee.com%2Fapi%2Fv5%2Frepos%2FVEXLife%2FAccelerated-TD&logo=gitee&style=social)

# Introduction

## Agents

`PlainATDAgent` updates ![](https://latex.codecogs.com/svg.image?\mathbf{A}) directly while `SVDATDAgent` and `DiagonalizedSVDATDAgent` update its singular value decompositions respectively which is thought to have a fewer complexity. The difference between `SVDATDAgent` and `DiagonalizedSVDATDAgent` is that `SVDATDAgent` employs the method mentioned here: [Brand 2006](https://pdf.sciencedirectassets.com/271586/1-s2.0-S0024379506X04573/1-s2.0-S0024379505003812/main.pdf), while `DiagonalizedSVDATDAgent` adopted the method mentioned here: [Gahring 2015](https://arxiv.org/pdf/1511.08495) which diagonalizes ![](https://latex.codecogs.com/svg.image?\mathbf{\Sigma}) so that the pseudo-inverse of the matrix is more easy to calculate though I still can't figure out completely how it works.

I also implemented a conventional Gradient Temporal Difference agent called `TDAgent`. I tested them in several environments as introduced below.

## Backend Support

I provided the backend support for PyTorch(CPU) to skip the process converting from `numpy.ndarray` to `torch.Tensor` and vice versa. You can achieve this by adding this code before importing `atd` module:
```python
import os
os.environ["ATD_BACKEND"] = "NumPy"  # or "PyTorch"
```

To test it yourself, just clone the repository and run `python algorithm_test/<random_walk or boyans_chain>.py`. :)

# Requirements

- Python>=3.9
- NumPy>=1.19
- Torch>=1.10 if you want to use PyTorch as backend
- Matplotlib>=3.3.3 if you want to run my test script
- Tqdm if you want to run my test script

# Tests

## Random Walk

This environment is from [Sutton's book](http://incompleteideas.net/book/RLbook2020.pdf).

The code file is [this](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/random_walk.py) and the result is [here](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/random_walk.png):
![random_walk](https://user-images.githubusercontent.com/36587232/144394107-d0cf9bd0-ea09-4e48-ba04-cb07af9e4240.png)

## Boyan's Chain

The environment was proposed in [Boyan 1999](https://www.researchgate.net/publication/2621189_Least-Squares_Temporal_Difference_Learning).

The code file is [this](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/boyans_chain.py) and the result is [here](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/boyans_chain.png):
![boyans_chain](https://user-images.githubusercontent.com/36587232/144394100-dc8c48c2-1d38-433f-aea6-f202da3bbb13.png)

# Usage

To import my implementation of the algorithm into your project, follow these instructions if you aren't very familiar with this.
1. Clone the repository and copy the `atd.py` to where you want. If you downloaded a .zip file from GitHub, remember to unzip it.
2. Add this code to your Python script's head:
   ```python
   from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent  # or any agent you want
   ```
3. If the destination directory is not the same as where your main Python file is, you should use this code snippet instead of Step 2 to append the directory to the environment variable so that the Python interpreter could find it. Alternatively, you can refer to `importlib` provided by later Python.
   ```python
   import sys

   sys.path.append("<The directory where you placed atd.py>")
   from atd import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent  # or any agent you want
   ```
4. Initialize an agent like this and you are ready to use it!
   ```python
   agent = TDAgent(lr=0.01, lambd=0, observation_space_n=4, action_space_n=2)
   ```

Reference: [Gahring 2016](https://arxiv.org/pdf/1611.09328.pdf)

Please feel free to make a Pull Request and I'm expecting your Issues.
