import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent, ActorCriticAgent

env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

# base = "models/random_ball"
# paths = []
# for root, dirs, files in os.walk(base):
#     for file in files:
#         if file.endswith(".pt"):
#             paths.append(f"{root}/{file}")

paths = [
    "models/random_ball/1/-0.01_every_to_1 iteration 2_31275_0.99.pt",
    "models/random_ball/2/random_ball_every_-1_goal_300_transfer_14705_299.pt",
    "models/random_ball/3/random_ball_normalized_5019_0.99.pt",
    "models/random_ball/4/random_ball_normalized_transfer_26914_299.pt"
]

sim_scores = []

for path in paths:
    agent.model = torch.jit.load(path, map_location="cpu")

    ep_rewards = []
    for _ in range(1000):
        rewards = []
        state = env.reset(ball_random=True)
        while not env.loop_once():
            state, reward, done = env.step(agent.policy(state))
            rewards.append(reward)
        ep_rewards.append(np.sum(rewards))

    sim_scores.append(np.mean(ep_rewards))

best_model = paths[np.argmax(sim_scores)]
print(best_model)
print(sim_scores)
plt.bar(paths, sim_scores)
plt.show()


# agent.model = torch.jit.load(f"models/random_ball/first/-0.01_every_to_1 iteration 2_31275_0.99.pt", map_location="cpu")

# ep_rewards = []
# for _ in range(100):
#     rewards = []
#     state = env.reset(ball_random=True)
#     while not env.loop_once():
#         state, reward, done = env.step(agent.policy(state))
#         rewards.append(reward)
#     ep_rewards.append(np.sum(rewards))

# print(np.mean(ep_rewards))
# plt.plot(ep_rewards)
# plt.show()
