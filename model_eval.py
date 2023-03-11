import torch
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent


env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

agent.model = torch.jit.load("best_model_dqn/-0.01_every_to_1_11029.pt", map_location="cpu")

while env.running:
    state = env.reset()
    while not env.loop_once():
        state, reward, done = env.step(agent.policy(state))
