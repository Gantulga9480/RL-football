import torch
import numpy as np
from test_env import TestEnv, STATE_SPACE_SIZE, ACTION_SPACE_SIZE
from RL import DeepQNetworkAgent, ActorCriticAgent
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="pt file")
parser.add_argument("--rb", action="store_true", help="random ball")
parser.add_argument("--ac", action="store_true", help="Actor critic model")
args = parser.parse_args()

env = TestEnv(title="Player path", env_count=1, random_ball=args.rb)
if args.ac:
    agent = ActorCriticAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
else:
    agent = DeepQNetworkAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
agent.training = False

if isinstance(agent, DeepQNetworkAgent):
    agent.model = torch.jit.load(args.file, map_location="cpu")
else:
    agent.actor = torch.jit.load(args.file, map_location="cpu")

ep_rewards = []

for i in range(1):
    rewards = []
    state = env.reset()
    done = False
    while not done:
        state, reward, done = env.step(agent.policy(state))
        done = any(done)
        rewards.append(reward)
    ep_rewards.append(np.sum(rewards))
    if not env.running:
        break

# env.loop_forever()

plt.plot(env.envs[0].player_speed)
plt.show()
