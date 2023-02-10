from Game import Game, core
from football import Football, ACTIONS
import random


class Test(Game):

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 30
        self.team_size = 3
        self.set_window()
        self.set_title(self.title)
        self.football = Football(self.window, self.size, self.fps, self.team_size, False)

    def onRender(self):
        actions = [random.choice(ACTIONS) for _ in range(self.team_size)]
        self.window.fill((255, 255, 255))
        self.football.step(actions)


Test().loop_forever()
