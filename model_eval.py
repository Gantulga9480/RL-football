import torch
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent


env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

agent.model = torch.jit.load("best_model_dqn/dqn_-0.01_every_to_1_7000.pt")
agent.model.to(agent.device)

while env.running:
    state = env.football.reset()
    while not env.loop_once():
        state, reward, done = env.step(agent.policy(state))
