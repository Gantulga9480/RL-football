from Game import Game
from Game import core
from football import Football, NOOP


ACTION_SPACE_SIZE = 6
STATE_SPACE_SIZE = 8


class Playground(Football):

    def __init__(self, window, size, fps, team_size, full: bool = True) -> None:
        super().__init__(window, size, fps, team_size, full)
        self.counter = 0

    def step(self, actions: list = None):
        super().step(actions)
        self.counter += 1
        if self.counter == (self.fps * 10):
            self.done = True
        if self.teamRight.score:
            self.done = True
        ball_pos = self.plane.to_xy(self.ball.position())
        if ball_pos[0] > self.plane.x_max or ball_pos[0] < self.plane.x_min or \
                ball_pos[1] > self.plane.y_max or ball_pos[1] < self.plane.y_min:
            self.done = True
        if ball_pos[0] < -60:
            self.done = True
        if self.done and not self.teamRight.score:
            reward = -100
        else:
            if self.teamRight.score:
                reward = self.teamRight.score * 100
            else:
                reward = 0
        return self.get_state(), reward, self.done

    def get_state(self):
        ball_pos = self.plane.to_xy(self.ball.position())
        ball_dir = self.ball.direction()
        ball_vel = self.ball.speed()
        player_pos = self.plane.to_xy(self.players[0].position())
        dir_now = self.teamRight.players[0].direction()
        speed = self.players[0].speed()
        state = [ball_pos[0], ball_pos[1], ball_dir, ball_vel, player_pos[0], player_pos[1], dir_now, speed]
        return state

    def reset(self):
        self.counter = 0
        self.done = False
        self.ball.reset((300, 0))
        self.teamRight.reset()
        return self.get_state()


class SinglePlayerFootball(Game):

    def __init__(self, title: str = 'Single Agent train') -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 30
        self.set_window()
        self.set_title(title)
        self.env: Playground = None
        self.team_size = 1
        self.step_count = 0
        self.setup()

    def step(self, action: int = NOOP):
        self.step_count += 1
        return self.env.step([action])

    def setup(self):
        self.env = Playground(self.window, self.size, self.fps, self.team_size, False)

    def loop_once(self):
        super().loop_once()
        return self.env.done

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
                self.env.done = True
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering

    def onRender(self):
        self.window.fill((255, 255, 255))
        self.env.show()
