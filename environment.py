from Game import Game
from Game import core
from Game.graphic import CartesianPlane
from Game.physics import (DynamicPolygonBody,
                          FreePolygonBody, Body,
                          StaticRectangleBody)
from Game.physics import EnginePolygon
import numpy as np


TEAM_COLOR = [(0, 162, 232), (34, 177, 76)]
TEAM_LEFT = 0
TEAM_RIGHT = 1
TEAMS = [TEAM_LEFT, TEAM_RIGHT]


class Ball(FreePolygonBody):

    def __init__(self,
                 id: int,
                 plane: CartesianPlane,
                 size: tuple,
                 max_speed: float = 0,
                 drag_coef: float = 0) -> None:
        super().__init__(id, plane, size, max_speed, drag_coef)
        self.is_free = True

    def show(self, vertex: bool = False, velocity: bool = False) -> None:
        self.step()
        core.draw.circle(self.shape.plane.window, (255, 0, 0), self.velocity.TAIL, self.radius)
        super().show(vertex, velocity)


class Player(DynamicPolygonBody):

    PLAYER_SIZE = 10
    PLAYER_MAX_SPEED = 2.5
    PLAYER_SPEED_BALL = 2
    PLAYER_MAX_TURN_RATE = 3
    PLAYER_MAX_FOV = 120

    def __init__(self,
                 id: int,
                 team_id: int,
                 plane: CartesianPlane,
                 size: tuple,
                 max_speed: float = 1,
                 drag_coef: float = 0.01,
                 friction_coef: float = 0.3) -> None:
        super().__init__(id, plane, size, max_speed, drag_coef, friction_coef)
        self.team_id = team_id
        self.kicked = False
        self.has_ball = False

    def kick(self, ball: Ball, power: float):
        if self.has_ball:
            power += 1
            d = self.velocity.dir()
            ball.velocity.head = (power * np.cos(d), power * np.sin(d))
            pos = ball.shape.plane.get_parent_vector().plane.to_xy(self.shape.plane.CENTER)
            ball.shape.plane.get_parent_vector().head = pos
            ball.is_free = True
            self.kicked = True
            self.has_ball = False

    def show(self, vertex: bool = False, velocity: bool = False) -> None:
        core.draw.circle(self.shape.plane.window, TEAM_COLOR[self.team_id], self.velocity.TAIL, self.radius)
        super().show(vertex, velocity)
        if self.has_ball:
            core.draw.circle(self.shape.plane.window, (255, 0, 0), self.velocity.TAIL, 3)
            core.draw.circle(self.shape.plane.window, (0, 0, 0), self.velocity.TAIL, 3, 1)


class Team:

    def __init__(self, team_id: int, team_size: int, plane: CartesianPlane, player_buffer: 'list[Player]') -> None:
        self.score = 0
        self.team_id = team_id
        self.team_size = team_size
        self.players: list[Player] = []
        if self.team_id == TEAM_LEFT:
            self.plane = plane.createPlane(100 - plane.CENTER[0], 0)
            for i in range(self.team_size):
                player_plane = self.plane.createPlane(400 + i * Player.PLAYER_SIZE * 2, 0)
                p = Player(player_buffer.__len__() + i, self.team_id, player_plane, (Player.PLAYER_SIZE,) * 5, Player.PLAYER_MAX_SPEED)
                self.players.append(p)
                player_buffer.append(p)
        elif self.team_id == TEAM_RIGHT:
            self.plane = plane.createPlane(plane.CENTER[0] - 100, 0)
            for i in range(self.team_size):
                player_plane = self.plane.createPlane(-400 + i * Player.PLAYER_SIZE * 2, 0)
                p = Player(player_buffer.__len__() + i, self.team_id, player_plane, (Player.PLAYER_SIZE,) * 5, Player.PLAYER_MAX_SPEED)
                self.players.append(p)
                player_buffer.append(p)

    def show(self):
        if self.team_id == TEAM_LEFT:
            core.draw.rect(self.plane.window, TEAM_COLOR[TEAM_LEFT], (11 + 1, 470 + 1, 50 - 2, 140 - 2))
        elif self.team_id == TEAM_RIGHT:
            core.draw.rect(self.plane.window, TEAM_COLOR[TEAM_RIGHT], (1860, 470 + 1, 50 - 2, 140 - 2))
        self.plane.show()


class Playground(Game):

    TEAM_SIZE = 5  # in one team
    BALL_SIZE = 10

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 120
        self.set_window()
        self.set_title(self.title)

        self.teams: list[Team] = []
        self.players: list[Player] = []
        self.ball: Ball = None
        self.current_player = -1
        self.last_player = -1

        self.bodies: list[Body] = []

    def setup(self):
        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.create_wall()

        for team in TEAMS:
            self.teams.append(Team(team, self.TEAM_SIZE, self.plane, self.players))

        for player in self.players:
            self.bodies.append(player)

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

        self.ball = Ball(0, self.plane.createPlane(0, 0), (self.BALL_SIZE,) * 10, drag_coef=0.01)

    def loop(self):
        self.check_ball()

        speed = self.players[0].speed()
        if self.keys[core.K_UP]:
            self.players[self.current_player].accelerate(5)
        if self.keys[core.K_DOWN]:
            self.players[self.current_player].accelerate(-1)
        if self.keys[core.K_LEFT]:
            self.players[self.current_player].rotate(Player.PLAYER_MAX_TURN_RATE / (speed + 1))
        if self.keys[core.K_RIGHT]:
            self.players[self.current_player].rotate(-Player.PLAYER_MAX_TURN_RATE / (speed + 1))

        # for player in self.players:
        #     r = np.random.random() * 2 - 5
        #     r1 = np.random.random() * 20 - 10
        #     player.accelerate(r1)
        #     player.rotate(r)

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
            if event.key == core.K_f:
                if self.current_player != -1:
                    self.players[self.current_player].kick(self.ball, 5)
                    self.players[self.current_player].velocity.max = Player.PLAYER_MAX_SPEED
                    self.last_player = self.current_player
                    # self.current_player = -1

    def onRender(self):
        width = 1
        self.window.fill((255,) * 3)
        core.draw.rect(self.window, (0,) * 3, (60, 90, 1800, 900), width)  # Touch line
        core.draw.rect(self.window, (0,) * 3, (60, 367, 100, 346), width)  # Left goal area
        core.draw.rect(self.window, (0,) * 3, (1760, 367, 100, 346), width)  # Right goal area
        core.draw.rect(self.window, (0,) * 3, (60, 149, 321, 783), width)  # Left penalty area
        core.draw.rect(self.window, (0,) * 3, (1539, 149, 321, 783), width)  # Right penalty area
        core.draw.circle(self.window, (0, 0, 0), (960, 540), 180, width)
        core.draw.line(self.window, (0,) * 3, (960, 90), (960, 989))
        core.draw.rect(self.window, (0, 0, 0), (11, 470, 50, 140), width)
        core.draw.rect(self.window, (0, 0, 0), (1859, 470, 50, 140), width)
        self.plane.show()
        for team in self.teams:
            team.show()
        self.engine.step()
        if self.ball.is_free:
            self.ball.show()
        p1 = self.players[self.current_player]
        p1_plane = p1.shape.plane
        unit = p1.velocity.unit(100, vector=False)
        vv1 = p1_plane.createVector(unit[0], unit[1])
        vv2 = p1_plane.createVector(unit[0], unit[1])
        vv1.rotate(Player.PLAYER_MAX_FOV / 360 * np.pi)
        vv2.rotate(-Player.PLAYER_MAX_FOV / 360 * np.pi)
        vv1.show()
        vv2.show()
        for player in self.players[1:]:
            if player != p1:
                pos = player.shape.plane.CENTER
                pos = p1_plane.to_xy(pos)
                v = p1_plane.createVector(pos[0], pos[1])
                ab = np.arccos(p1.velocity.dot(v) / (p1.velocity.mag() * v.mag())) / np.pi * 180
                if ab < Player.PLAYER_MAX_FOV / 2:
                    v.show()

    def check_ball(self):
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
                    player.kicked = False
                    player.has_ball = True
                    player.velocity.max = Player.PLAYER_SPEED_BALL
                    self.current_player = p_idx
            if self.ball.is_free:
                pos = self.ball.position()
                # Left team scored a goal
                if (pos[0] < self.plane.x_min + 60 and -70 < pos[1] < 70):
                    self.ball.velocity.head = (1, 0)
                    self.ball.shape.plane.get_parent_vector().head = (0, 0)
                    self.teams[TEAM_LEFT].score += 1
                # Right team scored a goal
                elif (pos[0] > self.plane.x_max - 60 and -70 < pos[1] < 70):
                    self.ball.velocity.head = (1, 0)
                    self.ball.shape.plane.get_parent_vector().head = (0, 0)
                    self.teams[TEAM_RIGHT].score += 1

    def create_wall(self, wall_width=120, wall_height=5):
        y = self.size[1] // 2 - wall_width // 2 - wall_height // 2
        for _ in range(self.size[1] // wall_width):
            vec = self.plane.createVector(-self.size[0] // 2, y)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width), vec),
                                    (wall_height, wall_width)))
            vec = self.plane.createVector(self.size[0] // 2, y)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (wall_width, wall_width), vec),
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
