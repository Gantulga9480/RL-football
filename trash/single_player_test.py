from single_player_env_parallel import SinglePlayer, ACTIONS, STATE_SPACE_SIZE
from RL.dqn import DQN
import numpy as np

model = DQN(ACTIONS, 0.001, 0.99, gpu=False)
model.load_model("model/2023-02-09/model-450000.h5")

sim = SinglePlayer(1)

states = []
for env in sim.envs:
    states.append(env.reset())

while sim.running:
    for i in range(1):
        if sim.envs[i].done:
            states[i] = sim.envs[i].reset()

    actions = model.policy(states, greedy=True)

    infos = sim.step(actions)

    for i, info in enumerate(infos):
        new_state, reward, done = info
        states[i] = new_state
