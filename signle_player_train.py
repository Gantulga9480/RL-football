from single_player_env import SinglePlayer, ACTIONS, STATE_SPACE_SIZE
from RL.dqn import DQN


MAX_REPLAY_BUFFER = 240000
BATCH_SIZE = 128
EPOCHS = 2
TARGET_NET_UPDATE_FREQ = 300
MAIN_NET_TRAIN_FREQ = 30
CURRENT_TRAIN_ID = '2023-01-26'
ENV_COUNT = 10

model = DQN(ACTIONS, 0.001, 0.99, gpu=True)
model.create_model([STATE_SPACE_SIZE, 20, 20, ACTIONS.__len__()],
                   epochs=EPOCHS,
                   batchs=BATCH_SIZE,
                   train_freq=MAIN_NET_TRAIN_FREQ,
                   update_freq=TARGET_NET_UPDATE_FREQ)
model.create_buffer(MAX_REPLAY_BUFFER, model.batchs * 10)

sim = SinglePlayer(ENV_COUNT)

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

    info = ' '.join([
        f'e: {model.e}',
        f'r: {round(sum(rewards) / ENV_COUNT, 2)}'
    ])
    print(info)

print(sim.step_count)
# save trained model
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model'])
# model.save_model(path)

# save training reward history
# with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
#     f.write('\n'.join([str(item) for item in avg_reward_hist]))
