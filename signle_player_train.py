from single_player_env import SinglePlayer
from RL.dqn import DQN, ReplayBuffer
# from model import DQN, ReplayBuffer


MAX_REPLAY_BUFFER = 24000
TARGET_NET_UPDATE_FREQ = 300
MAIN_NET_TRAIN_FREQ = 30
CURRENT_TRAIN_ID = '2023-01-09'
ENV_COUNT = 5

model = DQN(0.001, 0.99, 128, 1, False)
model.create_model([7, 14, 5])
# model.train = True
# model.e = 0

replay_buffer = ReplayBuffer(MAX_REPLAY_BUFFER, 256)

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
        n_state, reward, done = info
        state = states[i]
        action = actions[i]
        replay_buffer.push([state, action, n_state, reward, done])
        states[i] = n_state
        rewards.append(reward)

    if replay_buffer.trainable and model.train:
        if sim.step_count % MAIN_NET_TRAIN_FREQ == 0:
            model.learn(replay_buffer.sample(model.batchs))
        model.decay_epsilon(0.99995)
        if model.e == model.e_min:
            model.e = 0.2
        if sim.step_count % TARGET_NET_UPDATE_FREQ == 0:
            model.update_target()

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
