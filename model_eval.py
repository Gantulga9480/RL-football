import os
import platform
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent, ActorCriticAgent
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rb", action="store_true", help="random ball")
parser.add_argument("--ne", type=int, default=1000, help="Number of episodes")
args = parser.parse_args()

font = {'size': 18}
matplotlib.rc('font', **font)

env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

base = r"best_models"
paths = []
for root, dirs, files in os.walk(base):
    for file in files:
        if file.endswith(".pt"):
            if platform.system() == "Linux":
                paths.append(f"{root}/{file}")
            else:
                paths.append(f"{root}\\{file}")
paths.sort()
sim_scores = []
for path in paths:
    agent.model = torch.jit.load(path, map_location="cpu")
    ep_rewards = []
    env.set_title(path)
    for _ in range(args.ne):
        rewards = []
        state = env.reset(random_ball=args.rb)
        while not env.loop_once():
            state, reward, done = env.step(agent.policy(state))
            rewards.append(reward)
        ep_rewards.append(np.sum(rewards))
        if not env.running:
            break
    sim_scores.append(np.mean(ep_rewards))

best_model = paths[np.argmax(sim_scores)]
for n, s in zip(paths, sim_scores):
    print(f"{n}: {s}")
print(f"\n{best_model=}")

fig, ax = plt.subplots(1)
if len(sim_scores) > 1:
    plt.title("Average score of 1000 episodes")
    ax.set_ylabel("Score")
    ax.bar(list(range(len(sim_scores))), sim_scores)
    ax.set_xticks([])
else:
    plt.title("Model performance over 1000 episodes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.plot(ep_rewards)
plt.show()
