from single_agent_env import SinglePlayerFootball, ACTION_SPACE_SIZE, STATE_SPACE_SIZE
from RL.dqn import DQNAgent
from RL.utils import ReplayBuffer
import numpy as np
from torch import nn


class DQN(nn.Module):

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_shape)
        )

    def forward(self, x):
        return self.model(x)


MAX_REPLAY_BUFFER = 1_000_000
BATCH_SIZE = 128
TARGET_NET_UPDATE_FREQ = 100_000
MAIN_NET_TRAIN_FREQ = 1
ACTION_REPEAT = 1
SAVE_INTERVAL = 100_000
CURRENT_TRAIN_ID = f'2023-02-13/single-{BATCH_SIZE}-{TARGET_NET_UPDATE_FREQ}-{ACTION_REPEAT}'


sim = SinglePlayerFootball(CURRENT_TRAIN_ID)

model = DQNAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE, 0.003, 0.99, 0.9999999, device="cuda:0")
model.create_model(DQN(STATE_SPACE_SIZE, ACTION_SPACE_SIZE),
                   DQN(STATE_SPACE_SIZE, ACTION_SPACE_SIZE),
                   batchs=BATCH_SIZE,
                   train_freq=MAIN_NET_TRAIN_FREQ,
                   update_freq=TARGET_NET_UPDATE_FREQ)
model.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, 100_000))
model.e_min = 0.1


avg_rewards = []
action = None

while sim.running:
    episode_rewards = []
    state = sim.env.reset()
    while not sim.loop_once():
        if sim.step_count % ACTION_REPEAT == 0:
            action = model.policy(state)
        n_state, reward, done = sim.step(action)
        if sim.step_count % ACTION_REPEAT == 0:
            model.learn(state, action, n_state, reward, done)

        episode_rewards.append(reward)

        if sim.step_count % SAVE_INTERVAL == 0 and model.e < 0.5:
            path = '/'.join(['model', CURRENT_TRAIN_ID, f'model-{sim.step_count}-{round(model.e, 4)}.pt'])
            model.save_model(path)
        if sim.step_count % 10 == 0:
            for i in range(10):
                print(model.buffer.buffer.pop())
            quit()

    avg_rewards.append(np.mean(episode_rewards))
    print(' * '.join([f'e: {round(model.e, 4)}',
                      f'r: {round(np.mean(episode_rewards), 2)}']))

print(sim.step_count)

# save trained model
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model.pt'])
model.save_model(path)

with open('/'.join(['model', CURRENT_TRAIN_ID, 'model_info.txt']), 'w') as f:
    f.write(model.model.__repr__())

# save training reward history
with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
    f.write('\n'.join([str(item) for item in avg_rewards]))
