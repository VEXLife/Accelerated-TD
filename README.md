# Accelerated-TD

This is my implementation of the Accelerated Gradient Temporal Difference Learning algorithm (ATD) in Python.

`PlainATDAgent` updates ![](https://latex.codecogs.com/svg.image?\mathbf{A}) directly while `SVDATDAgent` and `DiagonalizedSVDATDAgent` update its singular value decompositions respectively which is thought to have a fewer complexity. The difference between `SVDATDAgent` and `DiagonalizedSVDATDAgent` is that `SVDATDAgent` employs the method mentioned here: [Brand 2006](https://pdf.sciencedirectassets.com/271586/1-s2.0-S0024379506X04573/1-s2.0-S0024379505003812/main.pdf), and `DiagonalizedSVDATDAgent` adopted the method mentioned here: [Gahring 2015](https://arxiv.org/pdf/1511.08495) which diagonalizes ![](https://latex.codecogs.com/svg.image?\mathbf{\Sigma}) so that the pseudo-inverse of the matrix is more easy to calculate though I still can't figure out completely how it works.
I also implemented a conventional Gradient Temporal Difference called `TDAgent`. I tested them in several environments as introduced below.

To test it yourself, just clone the repository and run `python algorithm_test/<random_walk or boyans_chain>.py`. :)

# Requirements

- Python>=3.8
- NumPy>=1.19
- Matplotlib>=3.3.3 if you want to run my test script
- Tqdm if you want to run my test script

# Tests

## Random Walk

This environment is from [Sutton's book](http://incompleteideas.net/book/RLbook2020.pdf).

The code file is [this](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/random_walk.py) and the result is [here](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/random_walk.png):
![random_walk](https://user-images.githubusercontent.com/36587232/141802348-08e20308-d5b2-4011-8655-3a89045f0eb5.png)

## Boyan's Chain

The environment was proposed by Boyan, but I don't have a direct link to his article.

The code file is [this](https://github.com/VEXLife/Accelerated-TD/blob/main/algorithm_test/boyans_chain.py) and the result is [here](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/boyans_chain.png):
![boyans_chain](https://user-images.githubusercontent.com/36587232/141802337-e345e59a-f856-49e7-b660-84dce14eea94.png)

# Usage

To import my implementation of the algorithm into your project, and you aren't very familiar with this, follow these instructions.
1. Clone the repository and copy the `ATD_cn.py` to where you want. If you downloaded a .zip file from GitHub, remember to unzip it.
2. Add these code to your Python script's head:
   ```python
   from ATD_cn import TDAgent, SVDATDAgent, SVDLRATDAgent, PlainATDAgent  # or any agent you want
   ```
3. If the destination directory is not the same as where your main Python file is, you should use this code snippet instead of Step 2 to append the directory to the environment variable so that the Python interpreter could find it. Alternatively, you can refer to `importlib` provided by later Python.
   ```python
   import sys

   sys.path.append("<The directory where you placed ATD_cn.py>")
   from ATD_cn import TDAgent, SVDATDAgent, SVDLRATDAgent, PlainATDAgent  # or any agent you want
   ```
4. Initialize an agent like this and you are ready to use it!
   ```python
   agent=TDAgent(lr=0.01, lambd=0, observation_space_n=4, action_space_n=2)
   ```

Reference: [Gahring 2016](https://arxiv.org/pdf/1611.09328.pdf)

Please feel free to make a Pull Request and I'm expecting your Issues.
