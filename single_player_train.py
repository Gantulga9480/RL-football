from single_player_env import SinglePlayer, ACTION_SPACE_SIZE, STATE_SPACE_SIZE
from RL.dqn import DQNAgent
from RL.utils import ReplayBuffer
import numpy as np
from torch import nn


class DQN(nn.Module):

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, output_shape)
        )

    def forward(self, x):
        return self.model(x)


MAX_REPLAY_BUFFER = 5000
BATCH_SIZE = 64
EPOCHS = 1
TARGET_NET_UPDATE_FREQ = 20
MAIN_NET_TRAIN_FREQ = 1
CURRENT_TRAIN_ID = f'2023-02-10-single-{BATCH_SIZE}-{TARGET_NET_UPDATE_FREQ}'
ENV_COUNT = 1
SAVE_INTERVAL = 10000

model = DQNAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE, 0.003, 0.99, 0.999999, device="cuda:0")
model.create_model(DQN(STATE_SPACE_SIZE, ACTION_SPACE_SIZE),
                   DQN(STATE_SPACE_SIZE, ACTION_SPACE_SIZE),
                   epochs=EPOCHS,
                   batchs=BATCH_SIZE,
                   train_freq=MAIN_NET_TRAIN_FREQ,
                   update_freq=TARGET_NET_UPDATE_FREQ)
model.create_buffer(ReplayBuffer(MAX_REPLAY_BUFFER, BATCH_SIZE * 10))
model.e_min = 0.1

sim = SinglePlayer(ENV_COUNT)

last_rewards = []
avg_rewards = []
states = []
for env in sim.envs:
    states.append(env.reset())

while sim.running:
    actions = []
    rewards = []
    current_states = []
    new_states = []
    dones = []
    for i in range(ENV_COUNT):
        if sim.envs[i].done:
            states[i] = sim.envs[i].reset()

    actions = model.policy(states)

    infos = sim.step(actions)

    for i, info in enumerate(infos):
        new_state, reward, done = info
        current_states.append(states[i])
        states[i] = new_state
        new_states.append(new_state)
        rewards.append(reward)
        dones.append(done)
    model.learn(current_states, actions, new_states, rewards, dones)

    last_rewards.append(np.sum(rewards) / ENV_COUNT)

    if sim.step_count % 100:
        info = ' '.join([
            f'e: {round(model.e, 4)}',
            f'r: {round(last_rewards[-1], 2)}',
            f'fps: {round(sim.clock.get_fps(), 2)}'
        ])
        print(info)

    if sim.step_count % SAVE_INTERVAL == 0:
        avg_rewards.append(np.sum(last_rewards) / SAVE_INTERVAL)
        path = '/'.join(['model', CURRENT_TRAIN_ID, f'model-{sim.step_count}-{round(model.e, 4)}.pt'])
        model.save_model(path)

print(sim.step_count)

# save trained model
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model.pt'])
model.save_model(path)

# save training reward history
with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
    f.write('\n'.join([str(item) for item in avg_rewards]))
