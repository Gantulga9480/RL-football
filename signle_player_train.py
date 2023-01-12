from single_player_env import SinglePlayer, ACTIONS, STATE_SPACE_SIZE
from RL.dqn import DQN, ReplayBuffer


MAX_REPLAY_BUFFER = 24000
TARGET_NET_UPDATE_FREQ = 300
MAIN_NET_TRAIN_FREQ = 30
CURRENT_TRAIN_ID = '2023-01-09'
ENV_COUNT = 5

model = DQN(ACTIONS, 0.001, 0.99)
model.create_model([STATE_SPACE_SIZE, 20, 20, ACTIONS.__len__()], epochs=3, gpu=True)
model.create_buffer(MAX_REPLAY_BUFFER, model.batchs)

sim = SinglePlayer(ENV_COUNT)

states = []
for env in sim.envs:
    states.append(env.reset())

while sim.running:
    actions = []
    rewards = []
    for i in range(ENV_COUNT):
        if sim.envs[i].done:
            states[i] = sim.envs[i].reset()

    for state in states:
        actions.append(model.policy(state))

    infos = sim.step(actions)

    for i, info in enumerate(infos):
        state = states[i]
        action = actions[i]
        n_state, reward, done = info
        model.learn(state, action, n_state, reward, done)
        states[i] = n_state
        rewards.append(reward)

    info = ' '.join([
        f'e: {model.e}',
        f'r: {sum(rewards) / ENV_COUNT}'
    ])
    # print(info)

# save trained model
path = '/'.join(['model', CURRENT_TRAIN_ID, 'model'])
model.save_model(path)
print(sim.step_count)

# save training reward history
# with open('/'.join(['model', CURRENT_TRAIN_ID, 'reward_hist.csv']), 'w') as f:
#     f.write('\n'.join([str(item) for item in avg_reward_hist]))
