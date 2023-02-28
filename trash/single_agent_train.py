import torch
import argparse
import numpy as np
from single_agent_env import SinglePlayerFootball, ACTION_SPACE_SIZE, STATE_SPACE_SIZE
from RL import DeepQNetworkAgent
from RL.utils import ReplayBuffer
from model import DQN

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-c", "--checkpoint", action="store_true")
args = parser.parse_args()


MAX_REPLAY_BUFFER = 1_000_000
BATCH_SIZE = 64
TARGET_NET_UPDATE_FREQ = 5
SAVE_INTERVAL = 10000
CURRENT_TRAIN_ID = f'2023-02-28'

torch.manual_seed(3407)
np.random.seed(3407)
sim = SinglePlayerFootball(title=CURRENT_TRAIN_ID)
model = DeepQNetworkAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE, device="cuda:0")
model.create_model(DQN, lr=0.00025, y=0.99, e_decay=0.999999, batchs=BATCH_SIZE, target_update_freq=TARGET_NET_UPDATE_FREQ)
model.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, 50_000, STATE_SPACE_SIZE))
model.train = args.train

'/home/sict/.local/bin'
avg_rewards = []
avg_eps_rewards = []
action = None

while sim.running:
    episode_rewards = []
    state = sim.env.reset()
    while not sim.loop_once():
        action = model.policy(state)
        n_state, reward, done = sim.step(action)
        model.learn(state, action, n_state, reward, done)

        state = n_state

        episode_rewards.append(reward)

        # if model.train_count % SAVE_INTERVAL == 0 and model.e < 0.3 and args.checkpoint:
        #     path = '/'.join(['model', CURRENT_TRAIN_ID, f'model-{sim.step_count}-e{round(model.e, 4)}-r{round(avg_rewards[-1], 2)}.pt'])
        #     model.save_model(path)

    avg_eps_rewards.append(np.sum(episode_rewards))
    if avg_eps_rewards.__len__() == 100:
        avg_rewards.append(np.sum(avg_eps_rewards) / 100)
        avg_eps_rewards = []
    print(' * '.join([f'e: {round(model.e, 4)}', f'r: {round(np.sum(episode_rewards), 2)}']))

print(model.step_count, model.train_count)

if args.checkpoint:
    # save trained model
    path = '/'.join(['model', CURRENT_TRAIN_ID, 'model.pt'])
    model.save_model(path)

    # save model structure to txt
    with open('/'.join(['model', CURRENT_TRAIN_ID, 'model_info.txt']), 'w') as f:
        f.write(model.model.__repr__())

    # save training reward history to csv
    with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
        f.write('\n'.join([str(item) for item in avg_rewards]))
