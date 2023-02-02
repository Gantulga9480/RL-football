from Game.graphic import CartesianPlane
from Game.physics import (Player, Ball, Body,
                          StaticRectangleBody)
from Game.physics import EnginePolygon
from Game import core
import numpy as np


TEAM_LEFT = 0
TEAM_RIGHT = 1

# Actions
STOP = 0
KICK = 1
GO_FORWARD = 2
TURN_LEFT = 3
TURN_RIGHT = 4
ACTIONS = [STOP, KICK, GO_FORWARD, TURN_LEFT, TURN_RIGHT]


class TeamLeft:

    TEAM_ID = TEAM_LEFT

    def __init__(self, team_size: int, plane: CartesianPlane) -> None:
        self.score = 0
        self.players: list[Player] = []
        self.team_size = team_size
        self.plane = plane.createPlane(100 - plane.CENTER[0], 0)
        for i in range(self.team_size):
            self.players.append(
                Player(100 + i,
                       self.TEAM_ID,
                       self.plane.createPlane(400 + i * Player.PLAYER_SIZE * 2, 0),
                       (Player.PLAYER_SIZE,) * 5,
                       Player.PLAYER_MAX_SPEED))

    def reset(self):
        self.score = 0

    def show(self):
        core.draw.rect(self.plane.window, Player.TEAM_COLOR[self.TEAM_ID], (11 + 1, 470 + 1, 50 - 2, 140 - 2))


class TeamRight:

    TEAM_ID = TEAM_RIGHT

    def __init__(self, team_size: int, plane: CartesianPlane) -> None:
        self.score = 0
        self.players: list[Player] = []
        self.team_size = team_size
        self.plane = plane.createPlane(plane.CENTER[0] - 100, 0)
        for i in range(self.team_size):
            self.players.append(
                Player(200 + i,
                       self.TEAM_ID,
                       self.plane.createPlane(-400 - i * Player.PLAYER_SIZE * 2, 0),
                       (Player.PLAYER_SIZE,) * 5,
                       Player.PLAYER_MAX_SPEED))

    def reset(self):
        self.score = 0

    def show(self):
        core.draw.rect(self.plane.window, Player.TEAM_COLOR[self.TEAM_ID], (1860, 470 + 1, 50 - 2, 140 - 2))


class Football:

    BALL_SIZE = 10

    GOAL_AREA_WIDTH = 120
    GOAL_AREA_HEIGHT = 400

    def __init__(self, window, size, fps, team_size, full: bool = True) -> None:
        self.window = window
        self.size = size
        self.fps = fps
        self.team_size = team_size

        self.players: list[Player] = []
        self.ball: Ball = None
        self.current_player = -1
        self.last_player = -1
        self.bodies: list[Body] = []
        self.done = False

        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.teamRight = TeamRight(self.team_size, self.plane)
        self.teamLeft = TeamLeft(self.team_size, self.plane)
        self.create_wall()

        for player in self.teamRight.players:
            self.players.append(player)
            self.bodies.append(player)
        if full:
            for player in self.teamLeft.players:
                self.players.append(player)
                self.bodies.append(player)

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

        self.ball = Ball(0, self.plane.createPlane(0, 0), (self.BALL_SIZE,) * 10, drag_coef=0.01)

    def step(self, actions: list = None):
        if actions:
            for i, action in enumerate(actions):
                speed = self.players[i].speed()
                if action == GO_FORWARD:
                    self.players[i].accelerate(5)
                elif action == STOP:
                    self.players[i].accelerate(-1)
                elif action == TURN_LEFT:
                    self.players[i].rotate(Player.PLAYER_MAX_TURN_RATE / (speed + 1))
                elif action == TURN_RIGHT:
                    self.players[i].rotate(-Player.PLAYER_MAX_TURN_RATE / (speed + 1))
        self.show()
        self.engine.step()
        self.ball.step()
        self.check_ball()

    def reset(self):
        self.done = False
        self.ball.reset((0, 0))
        self.teamLeft.reset()
        self.teamRight.reset()

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
        if (pos[0] < self.plane.x_min + self.GOAL_AREA_WIDTH) and (-self.GOAL_AREA_HEIGHT/2 < pos[1] < self.GOAL_AREA_HEIGHT/2):
            self.teamLeft.score += 1
        elif (pos[0] > self.plane.x_max - self.GOAL_AREA_WIDTH) and (-self.GOAL_AREA_HEIGHT/2 < pos[1] < self.GOAL_AREA_HEIGHT/2):
            self.teamRight.score += 1

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
        width = self.size[0] - self.GOAL_AREA_WIDTH * 2
        height = self.size[1] - 60 * 2
        core.draw.rect(self.window, (0, 0, 0), (Football.GOAL_AREA_WIDTH, 60, width, height), 1)  # Touch line
        core.draw.rect(self.plane.window,
                       Player.TEAM_COLOR[TEAM_LEFT],
                       (0, self.size[1] / 2 - self.GOAL_AREA_HEIGHT / 2, self.GOAL_AREA_WIDTH, self.GOAL_AREA_HEIGHT))
        core.draw.rect(self.plane.window,
                       (0, 0, 0),
                       (0, self.size[1] / 2 - self.GOAL_AREA_HEIGHT / 2, self.GOAL_AREA_WIDTH, self.GOAL_AREA_HEIGHT), 1)
        core.draw.rect(self.plane.window,
                       Player.TEAM_COLOR[TEAM_RIGHT],
                       (self.size[0] - self.GOAL_AREA_WIDTH, self.size[1] / 2 - self.GOAL_AREA_HEIGHT / 2, self.GOAL_AREA_WIDTH, self.GOAL_AREA_HEIGHT))
        core.draw.rect(self.plane.window,
                       (0, 0, 0),
                       (self.size[0] - self.GOAL_AREA_WIDTH, self.size[1] / 2 - self.GOAL_AREA_HEIGHT / 2, self.GOAL_AREA_WIDTH, self.GOAL_AREA_HEIGHT), 1)
        if self.ball.is_free:
            self.ball.show()
