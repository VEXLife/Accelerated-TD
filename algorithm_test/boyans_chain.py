# coding=utf-8
# Author: BWLL
# The Boyan's Chain Environment to test the algorithm

from ATD_cn import TDAgent, SVDATDAgent, DiagonalizedSVDATDAgent, PlainATDAgent
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

observations = [np.max(
    ((1 - np.abs(12 - 4 * np.arange(4) - N) / 4), np.zeros(4)), axis=0) for N in range(13)]
rng = np.random.default_rng()

w_optimal = np.arange(start=-24, stop=8, step=8)


def play_game(agent, timesteps=1000, iterations=100):
    records = []
    for _ in trange(iterations):
        time_step = 0
        episode = 0
        agent.reinit()
        agent.w *= 0
        record = []

        while time_step <= timesteps:
            pos = 12
            observation = observations[pos]

            while time_step <= timesteps:
                record.append(np.sqrt(np.mean((agent.w - w_optimal) ** 2)))
                pos -= rng.choice([1, 2]) if pos > 1 else 1
                next_observation = observations[pos]
                time_step += 1

                if pos == 0:
                    agent.learn(observation, next_observation, -2, 0, time_step)
                    episode += 1
                    break

                agent.learn(observation, next_observation, -3, 1, time_step)
                observation = next_observation

        records.append(record)
    return np.mean(np.array(records), axis=0)


plt.figure(dpi=120, figsize=(8, 6))

plt.plot(play_game(agent=TDAgent(lr=0.1, lambd=0, observation_space_n=4, action_space_n=2),
                   iterations=1), label="TD(0), $\\alpha=0.1$")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(alpha=1, k=30, eta=1e-4, lambd=0, observation_space_n=4,
                                                 action_space_n=2),
                   iterations=1),
         label="DiagonalizedSVDATD(0), $\\alpha=1$, \n$\\eta=1\\times10^{-4}$, $r=50$, Accuracy First")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(alpha=1, k=30, eta=1e-4, lambd=0, observation_space_n=4,
                                                 action_space_n=2, svd_diagonalizing=False,
                                                 w_update_emphasizes="complexity"),
                   iterations=1),
         label="DiagonalizedSVDATD(0), $\\alpha=0.1$, \n$\\eta=1\\times10^{-4}$, $r=50$, Complexity First")
plt.plot(play_game(agent=DiagonalizedSVDATDAgent(alpha=1, k=30, eta=1e-4, lambd=0, observation_space_n=4,
                                                 action_space_n=2, svd_diagonalizing=True,
                                                 w_update_emphasizes="complexity"),
                   iterations=1),
         label="DiagonalizedSVDATD(0), $\\alpha=0.1$, \n$\\eta=1\\times10^{-4}$, $r=50$, Complexity First\
          \nUsing SVD to diagonalize")
plt.plot(play_game(agent=SVDATDAgent(alpha=1, eta=1e-4, lambd=0, observation_space_n=4, action_space_n=2),
                   iterations=1), label="SVDATD(0), $\\alpha=1$, $\\eta=1\\times10^{-4}$")
plt.plot(play_game(agent=PlainATDAgent(alpha=1, eta=1e-4, lambd=0, observation_space_n=4, action_space_n=2),
                   iterations=1), label="PlainATD(0), $\\alpha=1$, $\\eta=1\\times10^{-4}$")
plt.legend()
plt.title("Boyan's Chain")
plt.xlabel("Timesteps")
plt.ylabel("RMSE")
plt.ylim(0, 14.5)
plt.xlim(0, 1000)
plt.savefig("./figures/boyans_chain.png", format="png")
