from single_player_env import SinglePlayer, ACTIONS
from model import DQN, ReplayBuffer
import numpy as np


MAX_REPLAY_BUFFER = 12000
TARGET_NET_UPDATE_FREQ = 18
MAIN_NET_TRAIN_FREQ = 3
CURRENT_TRAIN_ID = '2023-01-06'

model = DQN()
replay_buffer = ReplayBuffer(MAX_REPLAY_BUFFER, model.BATCH_SIZE)

env = SinglePlayer()
env.setup()

r_sum = 0
step_count = 0
episode_count = 0
last_avg_reward = 0
avg_reward_hist = []


while env.running:
    state = env.reset()
    while not env.done:
        action = model.predict_action(state)
        n_state, reward, done = env.step(action)

        r_sum += reward

        if done:
            episode_count += 1
            avg_reward_hist.append(r_sum)
            r_sum = 0

        replay_buffer.push([state, action, n_state, reward, done])
        state = n_state

        if replay_buffer.trainable:
            if step_count % MAIN_NET_TRAIN_FREQ == 0:
                model.train(replay_buffer.sample(model.BATCH_SIZE, 0.6))
            model.decay_epsilon()
            if model.epsilon == model.MIN_EPSILON:
                model.epsilon = 0.2
            if step_count % TARGET_NET_UPDATE_FREQ == 0:
                model.update_target()

        info = ' '.join([
            f'ep: {episode_count}',
            f'e: {model.epsilon}',
            f'r: {reward}'
        ])
        # print(info)

# save trained model
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model'])
model.save(path)

# save training reward history
with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
    f.write('\n'.join([str(item) for item in avg_reward_hist]))
