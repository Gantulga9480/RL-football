import os
import platform
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from single_agent_envs import SinglePlayerFootballParallel, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from RL import DeepQNetworkAgent, ActorCriticAgent
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", nargs='+', type=str, help="pt file")
parser.add_argument("--dir", type=str, help="Directory containing model pt file")
parser.add_argument("--ne", type=int, default=1000, help="Number of episodes")
parser.add_argument("--rb", action="store_true", help="random ball")
parser.add_argument("--ac", action="store_true", help="Actor critic model")
args = parser.parse_args()

font = {'size': 18}
matplotlib.rc('font', **font)
plt.style.use('ggplot')

env = SinglePlayerFootballParallel(title="Model evaluation", env_count=1, random_ball=args.rb)
if args.ac:
    agent = ActorCriticAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
else:
    agent = DeepQNetworkAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
agent.training = False

paths = []
if args.dir:
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".pt"):
                if platform.system() == "Linux":
                    paths.append(f"{root}/{file}")
                else:
                    paths.append(f"{root}\\{file}")
    paths.sort(key=lambda x: int(x.split('_')[-2]))
if args.file:
    for file in args.file:
        if file.endswith(".pt"):
            paths.append(file)
sim_scores = []
success_rate = []
for path in paths:
    if isinstance(agent, DeepQNetworkAgent):
        agent.model = torch.jit.load(path, map_location="cpu")
    else:
        agent.actor = torch.jit.load(path, map_location="cpu")
    ep_rewards = []
    successfull_play_count = 0
    env.set_title(path)
    for i in range(args.ne):
        rewards = []
        state = env.reset()
        done = False
        while not done:
            state, reward, done = env.step(agent.policy(state))
            done = any(done)
            rewards.append(reward)
            successfull_play_count += 1 if reward > 0 else 0
        ep_rewards.append(np.sum(rewards))
        if not env.running:
            break
    success_rate.append(successfull_play_count / (i + 1) * 100)
    sim_scores.append(np.mean(ep_rewards))

for n, s, r in zip(paths, sim_scores, success_rate):
    print(f"{n} - avg_score:{s}, success_rate:{r}")
best_scored_model = paths[np.argmax(sim_scores)]
best_success_rate_model = paths[np.argmax(success_rate)]
print(f"\n{best_scored_model=}: {np.max(sim_scores)}")
print(f"\n{best_success_rate_model=}: {np.max(success_rate)}")

fig, ax = plt.subplots(1)
if len(sim_scores) > 1:
    plt.title("Average score of 1000 episodes")
    ax.set_ylabel("Score")
    ax.bar(list(range(len(sim_scores))), sim_scores)
    ax.set_xticks([])
else:
    plt.title("Model performance over 1000 episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.plot(ep_rewards, c='r')
plt.show()
