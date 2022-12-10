from environment import Playground

env = Playground()
env.setup()

while env.running:
    env.loop_once()
