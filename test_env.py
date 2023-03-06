from Game import Game, core
from football import Football, STOP, GO_FORWARD, TURN_LEFT, TURN_RIGHT, KICK, NOOP


class Test(Game):

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 60
        self.team_size = 3
        self.set_window()
        self.set_title(self.title)
        self.football = Football(self.window, self.size, self.fps, self.team_size, True)

    def loop(self):
        actions = [NOOP for _ in range(self.team_size * 2 + 2)]  # +2 goal keeper agents
        idx = self.football.current_player
        if self.keys[core.K_UP]:
            actions[idx] = GO_FORWARD
        if self.keys[core.K_DOWN]:
            actions[idx] = STOP
        if self.keys[core.K_LEFT]:
            actions[idx] = TURN_LEFT
        if self.keys[core.K_RIGHT]:
            actions[idx] = TURN_RIGHT
        if self.keys[core.K_SPACE]:
            actions[idx] = KICK
        self.football.step(actions)
        if self.football.ball.is_out:
            self.football.ball.reset((0, 0))

    def onRender(self):
        self.window.fill((255, 255, 255))
        self.football.show()

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
            if event.key == core.K_a:
                if self.football.current_player < self.football.players.__len__() - 1:
                    self.football.current_player += 1
                else:
                    self.football.current_player = 0


Test().loop_forever()
