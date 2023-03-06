from Game.graphic import CartesianPlane
from Game.physics import (Player, Ball, Body,
                          StaticRectangleBody,
                          GoalKeeper)
from Game.physics import EnginePolygon
from Game import core
import numpy as np


TEAM_LEFT = 0
TEAM_RIGHT = 1
TEAM_ID_OFFSET = 100
TEAM_COLOR = [(0, 162, 232), (34, 177, 76)]

PLAYER_ABILITY_POINT = 0.98

BALL_SPEED_MAX = 20
BALL_SIZE = 20
GOAL_AREA_WIDTH = 200
GOAL_AREA_HEIGHT = 400

# Actions
NOOP = 0
STOP = 1
KICK = 2
GO_FORWARD = 3
TURN_LEFT = 4
TURN_RIGHT = 5
ACTIONS = [NOOP, STOP, KICK, GO_FORWARD, TURN_LEFT, TURN_RIGHT]


class TeamLeft:

    TEAM_ID = TEAM_LEFT

    def __init__(self, team_size: int, plane: CartesianPlane, goalkeeper: bool = True) -> None:
        self.score = 0
        self.players: list[Player] = []
        self.team_size = team_size
        self.plane = plane
        if goalkeeper:
            self.players.append(GoalKeeper(TEAM_LEFT * TEAM_ID_OFFSET, self.TEAM_ID, self.plane))
        for i in range(self.team_size):
            self.players.append(
                Player(TEAM_LEFT * TEAM_ID_OFFSET + i + 1, self.TEAM_ID, self.plane, ability_point=PLAYER_ABILITY_POINT))

    def reset(self):
        self.score = 0
        y_lim = (self.plane.window_size[1] - GOAL_AREA_WIDTH) / 2
        for player in self.players:
            x = np.random.randint(self.plane.x_min + GOAL_AREA_WIDTH, 0 + 1)
            y = np.random.randint(-y_lim, y_lim + 1)
            dr = np.random.random() * np.pi * 2
            player.reset((x, y), dr)


class TeamRight:

    TEAM_ID = TEAM_RIGHT

    def __init__(self, team_size: int, plane: CartesianPlane, goalkeeper: bool = True) -> None:
        self.score = 0
        self.players: list[Player] = []
        self.team_size = team_size
        self.plane = plane
        if goalkeeper:
            self.players.append(GoalKeeper(TEAM_RIGHT * TEAM_ID_OFFSET, self.TEAM_ID, self.plane))
        for i in range(self.team_size):
            self.players.append(
                Player(TEAM_RIGHT * TEAM_ID_OFFSET + i + 1, self.TEAM_ID, self.plane, ability_point=PLAYER_ABILITY_POINT))

    def reset(self):
        self.score = 0
        y_lim = (self.plane.window_size[1] - GOAL_AREA_WIDTH) / 2
        for player in self.players:
            x = np.random.randint(0, self.plane.x_max - GOAL_AREA_WIDTH + 1)
            y = np.random.randint(-y_lim, y_lim + 1)
            dr = np.random.random() * np.pi * 2
            player.reset((x, y), dr)


class Football:

    def __init__(self, window, size, fps, team_size, full: bool = True) -> None:
        self.window = window
        self.size = size
        self.fps = fps
        self.team_size = team_size

        self.players: list[Player] = []
        self.ball: Ball = None
        self.current_player = 0
        self.last_player = None
        self.bodies: list[Body] = []

        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.create_wall()
        self.teamRight = TeamRight(self.team_size, self.plane, goalkeeper=full)
        self.teamLeft = TeamLeft(self.team_size, self.plane, goalkeeper=full)

        for player in self.teamRight.players:
            self.players.append(player)
            self.bodies.append(player)
        if full:
            for player in self.teamLeft.players:
                self.players.append(player)
                self.bodies.append(player)

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

        self.ball = Ball(0, self.plane.createPlane(0, 0), (BALL_SIZE,) * 10, drag_coef=0.01)

        self.teamRight.reset()
        self.teamLeft.reset()

    def step(self, actions: list = None):
        if actions:
            for i, action in enumerate(actions):
                speed = self.players[i].speed() / 10
                if action == GO_FORWARD:
                    self.players[i].accelerate(10)
                elif action == STOP:
                    self.players[i].accelerate(-1)
                elif action == TURN_LEFT:
                    self.players[i].rotate(self.players[i].PLAYER_MAX_TURN_RATE / (speed + 1))
                elif action == TURN_RIGHT:
                    self.players[i].rotate(-self.players[i].PLAYER_MAX_TURN_RATE / (speed + 1))
                elif action == KICK:
                    self.players[i].kick(self.ball, BALL_SPEED_MAX)
                elif actions == NOOP:
                    pass
        self.engine.step()
        self.ball.step()
        self.check_ball()

    def reset(self):
        self.ball.reset((0, 0))
        self.teamLeft.reset()
        self.teamRight.reset()

    def check_ball(self):
        if not self.ball.is_out:
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
                        self.ball.is_free = False
                        self.ball.velocity.set_head_ref(player.velocity.get_head_ref())
                        self.ball.shape.plane.get_parent_vector().set_head_ref(player.shape.plane.get_parent_vector().get_head_ref())
                        player.has_ball = True
                        player.velocity.max = player.PLAYER_SPEED_BALL
                        self.current_player = p_idx
            if self.ball.is_free:
                pos = self.plane.to_xy(self.ball.position())
            else:
                if not tmp:
                    for i, p in enumerate(self.players):
                        if p.has_ball:
                            self.current_player = i
                            break
                pos = self.plane.to_xy(self.players[self.current_player].shape.plane.CENTER)
            if (pos[0] < self.plane.x_min + GOAL_AREA_WIDTH) and (-GOAL_AREA_HEIGHT / 2 < pos[1] < GOAL_AREA_HEIGHT / 2):
                self.teamLeft.score += 1
                self.players[self.current_player].has_ball = False
            elif (pos[0] > self.plane.x_max - GOAL_AREA_WIDTH) and (-GOAL_AREA_HEIGHT / 2 < pos[1] < GOAL_AREA_HEIGHT / 2):
                self.teamRight.score += 1
                self.players[self.current_player].has_ball = False

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
        width = self.size[0] - GOAL_AREA_WIDTH * 2
        height = self.size[1] - GOAL_AREA_WIDTH
        core.draw.rect(self.window, (0, 0, 0), (GOAL_AREA_WIDTH, GOAL_AREA_WIDTH / 2, width, height), 1)  # Touch line
        core.draw.circle(self.plane.window,
                         (0, 0, 0),
                         (0 + GOAL_AREA_WIDTH, self.size[1] / 2), 200, 1)
        core.draw.rect(self.plane.window,
                       TEAM_COLOR[TEAM_LEFT],
                       (0, self.size[1] / 2 - GOAL_AREA_HEIGHT / 2, GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT))
        core.draw.rect(self.plane.window,
                       (0, 0, 0),
                       (0, self.size[1] / 2 - GOAL_AREA_HEIGHT / 2, GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT), 1)
        core.draw.circle(self.plane.window,
                         (0, 0, 0),
                         (self.size[0] - GOAL_AREA_WIDTH, self.size[1] / 2), 200, 1)
        core.draw.rect(self.plane.window,
                       TEAM_COLOR[TEAM_RIGHT],
                       (self.size[0] - GOAL_AREA_WIDTH, self.size[1] / 2 - GOAL_AREA_HEIGHT / 2, GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT))
        core.draw.rect(self.plane.window,
                       (0, 0, 0),
                       (self.size[0] - GOAL_AREA_WIDTH, self.size[1] / 2 - GOAL_AREA_HEIGHT / 2, GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT), 1)
        for body in self.bodies:
            if isinstance(body, (Player, GoalKeeper)):
                body.show(TEAM_COLOR[body.team_id])
            else:
                body.show()
        p1 = self.players[self.current_player]
        p1_plane = p1.shape.plane
        unit = p1.velocity.unit(100, vector=False)
        vv1 = p1_plane.createVector(unit[0], unit[1])
        vv2 = p1_plane.createVector(unit[0], unit[1])
        vv1.rotate(p1.PLAYER_MAX_FOV / 360 * np.pi)
        vv2.rotate(-p1.PLAYER_MAX_FOV / 360 * np.pi)
        vv1.show()
        vv2.show()
        for player in self.players:
            if player != p1:
                pos = player.shape.plane.CENTER
                pos = p1_plane.to_xy(pos)
                v = p1_plane.createVector(pos[0], pos[1])
                dot = p1.velocity.mag() * v.mag()
                if dot > 0:
                    ab = np.arccos(p1.velocity.dot(v) / dot) / np.pi * 180
                    if ab < player.PLAYER_MAX_FOV / 2:
                        v.show()
        if self.ball.is_free:
            self.ball.show()
