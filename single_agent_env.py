from Game import Game
from Game import core
from Game.graphic import CartesianPlane
from Game.physics import StaticRectangleBody
from football import Football, NOOP, BALL_SPEED_MAX
import numpy as np


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
        ball_pos = self.plane.to_xy(self.ball.position())
        if self.ball.is_out or ball_pos[0] < -60:
            self.done = True
        if self.done and self.teamRight.score == 0:
            reward = -1
        else:
            if self.teamRight.score:
                self.done = True
                reward = 1
            else:
                reward = 0
        return self.get_state(), reward, self.done

    def get_state(self):
        ball_pos = self.plane.to_xy(self.ball.position())
        ball_dir = self.ball.direction() / 360
        ball_spd = self.ball.speed() / BALL_SPEED_MAX
        player_pos = self.plane.to_xy(self.players[0].position())
        player_dir = self.teamRight.players[0].direction() / 360
        player_spd = self.players[0].speed() / self.players[0].PLAYER_MAX_SPEED
        state = [ball_pos[0] / self.plane.x_max,
                 ball_pos[1] / self.plane.x_max, ball_dir, ball_spd,
                 player_pos[0] / self.plane.x_max,
                 player_pos[1] / self.plane.x_max, player_dir, player_spd]
        return np.array(state)

    def reset(self):
        self.counter = 0
        self.done = False
        self.ball.reset((300, 0))
        self.teamRight.reset()
        return self.get_state()

    def create_wall(self, wall_width=120, wall_height=5):
        y = self.size[1] // 2 - wall_width // 2 - wall_height // 2
        for _ in range(self.size[1] // wall_width):
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width),
                                                   self.plane.createVector(-60 // 2, y)),
                                    (wall_height, wall_width)))
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width),
                                                   self.plane.createVector(self.size[0] // 2, y)),
                                    (wall_height, wall_width)))
            y -= wall_width

        x = 0
        for _ in range(self.size[0] // wall_width):
            vec = self.plane.createVector(x, self.size[1] // 2)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width), vec),
                                    (wall_width, wall_height)))
            vec = self.plane.createVector(x, -self.size[1] // 2)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width), vec),
                                    (wall_width, wall_height)))
            x += wall_width


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

    def reset(self):
        return self.env.reset()

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
