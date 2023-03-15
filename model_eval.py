import torch
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent, ActorCriticAgent


env = SinglePlayerFootball(title="Model evaluation")
agent = ActorCriticAgent(None, None)
agent.train = False

agent.model = torch.jit.load("best_models/ac_fixed_ball_normalized_160_0.95.pt", map_location="cpu")

while env.running:
    state = env.reset()
    while not env.loop_once():
        state, reward, done = env.step(agent.policy(state))
