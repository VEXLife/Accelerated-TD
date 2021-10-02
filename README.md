# Accelerated-TD

This is my implementation of the Accelerated Gradient Temporal Difference Learning algorithm in Python.

`PlainATDAgent` updates ![](https://latex.codecogs.com/svg.image?\boldsymbol{A}) directly while `SVDATDAgent` and `SVDLRATDAgent` update its singular value decompositions respectively which is thought to have a fewer complexity. The difference between `SVDATDAgent` and `SVDLRATDAgent` is that `SVDATDAgent` employs the method mentioned here: [Brand 2006](https://pdf.sciencedirectassets.com/271586/1-s2.0-S0024379506X04573/1-s2.0-S0024379505003812/main.pdf), and `SVDLRATDAgent` adopted the method mentioned here: [Gahring 2015](https://arxiv.org/pdf/1511.08495) though I still can't figure out how it works.
I also implemented a conventional Gradient Temporal Difference called `TDAgent`. I tested them in the Random Walking Environment and the result is [here](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/random_walking.png):
![random_walking](https://user-images.githubusercontent.com/36587232/135722650-8f17ffdd-76e8-4991-a026-88cafc66bb75.png)

To test it yourself, just clone the repository and run `python algorithm_test/random_walk.py`. :)

# Requirements

- Python>=3.8
- NumPy>=1.19
- Matplotlib>=3.3.3 if you want to run my test script
- Tqdm if you want to run my test script

Reference: [Gahring 2016](https://arxiv.org/pdf/1611.09328.pdf)
