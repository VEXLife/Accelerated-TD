#!python
# -*- coding: utf-8 -*-
# @Author：BWLL
# The Random Walking Environment to test the algorithm

import sys

sys.path.append(".")

from atd_cn import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent, Backend
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

N = 7
v_true = Backend.arange(N) / (N - 1)
v = None
rng=np.random.default_rng()


def play_game(agent, episodes=100, iterations=100):
    global v

    records = []

    for _ in trange(iterations):
        record = []
        agent.reinit()
        agent.w = Backend.zeros(N)
        t = 0

        for i in range(episodes):
            pos = (N - 1) / 2
            observation = Backend.eye(N)[int((N - 1) / 2)]
            agent.reset()
            record.append(Backend.sqrt(Backend.mean((agent.w[1:N - 1] - v_true[1:N - 1]) ** 2)))

            while True:
                pos += rng.choice((-1, 1))
                next_observation = Backend.eye(N)[int(pos)]

                if pos == N - 1:
                    agent.learn(observation, next_observation, 1, 0, t)
                    break

                if pos == 0:
                    agent.learn(observation, next_observation, 0, 0, t)
                    break

                agent.learn(observation, next_observation, 0, 1, t)
                observation = next_observation
                t += 1

        records.append(record)
    return Backend.mean(Backend.create_matrix_func(records), 0)


plt.figure(dpi=120, figsize=(8, 6))

plt.plot(play_game(agent=TDAgent(lr=0.1, lambd=0.5, observation_space_n=7, action_space_n=2),
                   iterations=10, episodes=100), label="TD(0.5), $\\alpha=0.1$")
plt.plot(play_game(
    agent=DiagonalizedSVDATDAgent(k=30, eta=1e-4, lambd=0.5, observation_space_n=7, action_space_n=2),
    iterations=10, episodes=100),
    label="DiagonalizedSVDATD(0.5), $\\alpha=\\frac{1}{1+t}$, \n$\\eta=1\\times10^{-4}$, $r=30$, Accuracy First")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(k=30, eta=1e-4, lambd=0.5, observation_space_n=7,
                                                 action_space_n=2, svd_diagonalizing=False,
                                                 w_update_emphasizes="complexity"),
                   iterations=10, episodes=100),
         label="DiagonalizedSVDATD(0.5), $\\alpha=\\frac{1}{1+t}$, \n$\\eta=1\\times10^{-4}$, $r=30$, Complexity First")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(k=30, eta=1e-4, lambd=0.5, observation_space_n=7,
                                                 action_space_n=2, svd_diagonalizing=True,
                                                 w_update_emphasizes="complexity"),
                   iterations=10, episodes=100),
         label="DiagonalizedSVDATD(0.5), $\\alpha=\\frac{1}{1+t}$, \n$\\eta=1\\times10^{-4}$, $r=30$, Complexity First,\
         \nUsing SVD to diagonalize")
plt.plot(play_game(agent=SVDATDAgent(eta=1e-4, lambd=0.5, observation_space_n=7, action_space_n=2),
                   iterations=10, episodes=100),
         label="SVDATD(0.5), $\\alpha=\\frac{1}{1+t}$, $\\eta=1\\times10^{-4}$")
plt.plot(play_game(agent=PlainATDAgent(eta=1e-4, lambd=0.5, observation_space_n=7, action_space_n=2),
                   iterations=10, episodes=100),
         label="PlainATD(0.5), $\\alpha=\\frac{1}{1+t}$, $\\eta=1\\times10^{-4}$")
plt.legend()
plt.title("Random Walking")
plt.xlabel("Episode")
plt.ylabel("RMSE")
plt.savefig("./figures/random_walk.png", format="png")
