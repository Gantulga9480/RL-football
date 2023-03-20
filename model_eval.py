import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent, ActorCriticAgent

font = {'size': 18}
matplotlib.rc('font', **font)

env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

base = r"models\random_ball"
paths = []
for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith(".pt"):
            paths.append(f"{root}\\{file}")
paths.sort()

sim_scores = []
for path in paths:
    agent.model = torch.jit.load(path, map_location="cpu")
    ep_rewards = []
    for _ in range(10):
        rewards = []
        state = env.reset(ball_random=True)
        while not env.loop_once():
            state, reward, done = env.step(agent.policy(state))
            rewards.append(reward)
        ep_rewards.append(np.sum(rewards))
    sim_scores.append(np.mean(ep_rewards))

best_model = paths[np.argmax(sim_scores)]
print(f"{best_model=}")
print(sim_scores)

fig, ax = plt.subplots(1)
if len(sim_scores) > 1:
    plt.title("Average score of 1000 episodes")
    ax.set_xlabel("Trainig iterations")
    ax.set_ylabel("Score")
    ax.bar([range(len(sim_scores))], sim_scores)
    ax.set_xticks([])
else:
    plt.title("Model performance over 1000 episodes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.plot(ep_rewards)
plt.show()
