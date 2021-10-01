# Accelerated-TD

This is my implementation of the Accelerated Gradient Temporal Difference Learning algorithm in Python.

PlainATDAgent updates $\mathbf{A}$ directly while SVDATDAgent updates its singular value decompositions respectively which is thought to have a fewer complexity.
I also implemented a conventional Gradient Temporal Difference called TDAgent. I tested them in the Random Walking Environment and the result is [here](https://github.com/VEXLife/Accelerated-TD/blob/main/figures/random_walking.png):
![random_walking](https://user-images.githubusercontent.com/36587232/135572306-a48211e0-69fd-4fe9-8048-7414c011b643.png)

To test it yourself, just clone the repository and run `python algorithm_test/random_walk.py`

Reference: [Gahring 2016](https://arxiv.org/pdf/1611.09328.pdf)
