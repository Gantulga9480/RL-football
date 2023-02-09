from single_player_env import SinglePlayer, ACTIONS, STATE_SPACE_SIZE
from RL.dqn import DQN
import numpy as np

MAX_REPLAY_BUFFER = 50000
BATCH_SIZE = 256
EPOCHS = 1
TARGET_NET_UPDATE_FREQ = 50
MAIN_NET_TRAIN_FREQ = 10
CURRENT_TRAIN_ID = '2023-02-09'
ENV_COUNT = 50
SAVE_INTERVAL = 30000

model = DQN(ACTIONS, 0.001, 0.99, gpu=True)
model.create_model([STATE_SPACE_SIZE, 8, 16, 8, ACTIONS.__len__()],
                   epochs=EPOCHS,
                   batchs=BATCH_SIZE,
                   train_freq=MAIN_NET_TRAIN_FREQ,
                   update_freq=TARGET_NET_UPDATE_FREQ)
model.create_buffer(MAX_REPLAY_BUFFER, model.batchs * 10)
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

    info = ' '.join([
        f'e: {round(model.e, 4)}',
        f'r: {round(sum(rewards) / ENV_COUNT, 2)}',
        f'fps: {round(sim.clock.get_fps(), 2)}'
    ])
    print(info)

    if sim.step_count % SAVE_INTERVAL == 0:
        avg_rewards.append(np.sum(last_rewards) / SAVE_INTERVAL)
        path = '/'.join(['model', CURRENT_TRAIN_ID, f'model-{sim.step_count}'])
        model.save_model(path)

print(sim.step_count)
# save trained model
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model'])
model.save_model(path)

# save training reward history
with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
    f.write('\n'.join([str(item) for item in avg_rewards]))
