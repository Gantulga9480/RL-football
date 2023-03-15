import torch
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent


env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

agent.model = torch.jit.load("best_models/random_ball_every_-1_goal_300_transfer_14705_299.pt", map_location="cpu")

while env.running:
    state = env.reset(ball_random=True)
    while not env.loop_once():
        state, reward, done = env.step(agent.policy(state))
