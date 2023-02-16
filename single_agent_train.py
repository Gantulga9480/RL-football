from single_agent_env import SinglePlayerFootball, ACTION_SPACE_SIZE, STATE_SPACE_SIZE
from RL.dqn import DQNAgent
from RL.utils import ReplayBuffer
from model import DQN
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-c", "--checkpoint", action="store_true")
args = parser.parse_args()


MAX_REPLAY_BUFFER = 1_000_000
BATCH_SIZE = 1024
TARGET_NET_UPDATE_FREQ = 1000
SAVE_INTERVAL = 10000
CURRENT_TRAIN_ID = f'2023-02-15/single-b{BATCH_SIZE}-t{TARGET_NET_UPDATE_FREQ}-tanh'


sim = SinglePlayerFootball(CURRENT_TRAIN_ID)

model = DQNAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE, 0.003, 0.99, 0.999999, device="cuda:0")
model.create_model(DQN, batchs=BATCH_SIZE, target_update_freq=TARGET_NET_UPDATE_FREQ)
model.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, 50_000))
model.e_min = 0.1
model.train = args.train


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

        if model.train_count % SAVE_INTERVAL == 0 and model.e < 0.3 and args.checkpoint:
            path = '/'.join(['model', CURRENT_TRAIN_ID, f'model-{sim.step_count}-e{round(model.e, 4)}-r{round(avg_rewards[-1], 2)}.pt'])
            model.save_model(path)

    avg_eps_rewards.append(np.sum(episode_rewards))
    if avg_eps_rewards.__len__() == 100:
        avg_rewards.append(np.sum(avg_eps_rewards))
        avg_eps_rewards = []
    print(' * '.join([f'e: {round(model.e, 4)}', f'r: {round(np.sum(episode_rewards), 2)}']))

print(sim.step_count, model.train_count)

if args.save:
    # save trained model
    path = '/'.join(['model', CURRENT_TRAIN_ID, 'model.pt'])
    model.save_model(path)

    # save model structure to txt
    with open('/'.join(['model', CURRENT_TRAIN_ID, 'model_info.txt']), 'w') as f:
        f.write(model.model.__repr__())

    # save training reward history to csv
    with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
        f.write('\n'.join([str(item) for item in avg_rewards]))
