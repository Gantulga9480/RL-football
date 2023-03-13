import torch
from single_agent_env import SinglePlayerFootball
from RL import DeepQNetworkAgent


env = SinglePlayerFootball(title="Model evaluation")
agent = DeepQNetworkAgent(None, None)
agent.train = False

agent.model = torch.jit.load("random_ball/first/-0.01_every_to_1 iteration 2_31275_0.99.pt", map_location="cpu")

while env.running:
    state = env.reset(ball_random=True)
    while not env.loop_once():
        state, reward, done = env.step(agent.policy(state))
