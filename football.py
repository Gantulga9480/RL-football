from Game.graphic import CartesianPlane
from Game.physics import (Player, Ball, Body,
                          StaticRectangleBody)
from Game.physics import EnginePolygon
from Game import core
import numpy as np
import itertools


TEAM_LEFT = 0
TEAM_RIGHT = 1

# Actions
KICK = 0
GO_FORWARD = 1
STOP = 2
TURN_LEFT = 3
TURN_RIGHT = 4
ACTIONS = [KICK, GO_FORWARD, STOP, TURN_LEFT, TURN_RIGHT]


class TeamLeft:

    def __init__(self, team_size: int, plane: CartesianPlane) -> None:
        self.score = 0
        self.team_id = 0
        self.players: list[Player] = []
        self.team_size = team_size
        self.plane = plane.createPlane(100 - plane.CENTER[0], 0)
        for i in range(self.team_size):
            self.players.append(
                Player(100 + i,
                       self.team_id,
                       self.plane.createPlane(400 + i * Player.PLAYER_SIZE * 2, 0),
                       (Player.PLAYER_SIZE,) * 5,
                       Player.PLAYER_MAX_SPEED))

    def reset(self):
        self.score = 0

    def show(self):
        core.draw.rect(self.plane.window, Player.TEAM_COLOR[self.team_id], (11 + 1, 470 + 1, 50 - 2, 140 - 2))


class TeamRight:

    def __init__(self, team_size: int, plane: CartesianPlane, player_buffer: 'list[Player]') -> None:
        self.score = 0
        self.team_id = 1
        self.players: list[Player] = []
        self.team_size = team_size
        self.plane = plane.createPlane(plane.CENTER[0] - 100, 0)
        for i in range(self.team_size):
            self.players.append(
                Player(200 + i,
                       self.team_id,
                       self.plane.createPlane(-400 + i * Player.PLAYER_SIZE * 2, 0)
                       (Player.PLAYER_SIZE,) * 5,
                       Player.PLAYER_MAX_SPEED))

    def reset(self):
        self.score = 0

    def show(self):
        core.draw.rect(self.plane.window, Player.TEAM_COLOR[self.team_id], (1860, 470 + 1, 50 - 2, 140 - 2))


class Football:

    TEAM_SIZE = 10  # in one team
    BALL_SIZE = 10

    def __init__(self, window, size, fps) -> None:
        self.window = window
        self.size = size
        self.fps = fps

        self.players: list[Player] = []
        self.ball: Ball = None
        self.current_player = -1
        self.last_player = -1
        self.bodies: list[Body] = []
        self.done = False

        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.teamL = TeamLeft(self.TEAM_SIZE, self.plane)
        self.teamR = TeamRight(self.TEAM_SIZE, self.plane)
        self.create_wall()

        for player in itertools.chain(self.teamL.players, self.teamR.players):
            self.players.append(player)
            self.bodies.append(player)

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

        self.ball = Ball(0, self.plane.createPlane(650, 0), (self.BALL_SIZE,) * 10, drag_coef=0.01)

    def step(self):
        pass

    def reset(self):
        self.done = False
        self.ball.reset((0, 0))
        self.teamL.reset()
        self.teamR.reset()

    def check_ball(self):
        tmp = self.ball.is_free
        if self.ball.is_free:
            dists = []
            idx = []
            for i, player in enumerate(self.players):
                d = player.shape.plane.get_parent_vector().dist(self.ball.shape.plane.get_parent_vector())
                if (d <= (player.radius + self.ball.radius)):
                    dists.append(d)
                    idx.append(i)
                else:
                    player.kicked = False
            if idx.__len__() > 0:
                p_idx = idx[np.argmin(dists)]
                player = self.players[p_idx]
                if player.kicked:
                    pass
                else:
                    self.ball.velocity.head = (1, 0)
                    self.ball.is_free = False
                    player.has_ball = True
                    player.velocity.max = Player.PLAYER_SPEED_BALL
                    self.current_player = p_idx
        if self.ball.is_free:
            pos = self.ball.position()
        else:
            if not tmp:
                for i, p in enumerate(self.players):
                    if p.has_ball:
                        self.current_player = i
                        break
            pos = self.plane.to_xy(self.players[self.current_player].shape.plane.CENTER)
        if (pos[0] < self.plane.x_min + 60 and -70 < pos[1] < 70):
            self.teamL.score += 1
        elif (pos[0] > self.plane.x_max - 60 and -70 < pos[1] < 70):
            self.teamR.score += 1

    def create_wall(self, wall_width=120, wall_height=5):
        y = self.size[1] // 2 - wall_width // 2 - wall_height // 2
        for _ in range(self.size[1] // wall_width):
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width),
                                                   self.plane.createVector(-self.size[0] // 2, y)),
                                    (wall_height, wall_width)))
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width),
                                                   self.plane.createVector(self.size[0] // 2, y)),
                                    (wall_height, wall_width)))
            y -= wall_width

        x = -self.size[0] // 2 + wall_width // 2
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

    def show(self):
        self.draw_field()
        self.teams[TEAM_RIGHT].show()
        self.teams[TEAM_LEFT].show()
        if self.ball.is_free:
            self.ball.show()

    def draw_field(self, width=1):
        core.draw.rect(self.window, (0,) * 3, (60, 90, 1800, 900), width)  # Touch line
        core.draw.rect(self.window, (0,) * 3, (60, 367, 100, 346), width)  # Left goal area
        core.draw.rect(self.window, (0,) * 3, (1760, 367, 100, 346), width)  # Right goal area
        core.draw.rect(self.window, (0,) * 3, (60, 149, 321, 783), width)  # Left penalty area
        core.draw.rect(self.window, (0,) * 3, (1539, 149, 321, 783), width)  # Right penalty area
        core.draw.circle(self.window, (0, 0, 0), (960, 540), 180, width)
        core.draw.line(self.window, (0,) * 3, (960, 90), (960, 989))
        core.draw.rect(self.window, (0, 0, 0), (11, 470, 50, 140), width)
        core.draw.rect(self.window, (0, 0, 0), (1859, 470, 50, 140), width)
