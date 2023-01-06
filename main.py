from environment import Playground
from single_player_env import SinglePlayer, ACTIONS

# env = SinglePlayer()
env = Playground()
env.setup()

while env.running:
    env.loop_once()
