from Game import Game
from Game import core
from Game.graphic import CartesianPlane
from Game.physics import (DynamicPolygonBody,
                          FreePolygonBody, Body,
                          StaticRectangleBody)
from Game.physics import EnginePolygon
import numpy as np


TEAM_COLOR = [(0, 162, 232), (34, 177, 76)]
TEAM_BLUE = 0
TEAM_GREEN = 1


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
        power += 1
        d = self.velocity.dir()
        ball.velocity.head = (power*np.cos(d), power*np.sin(d))
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
        self.team_id = team_id
        self.team_size = team_size
        self.players: list[Player] = []
        if self.team_id == TEAM_BLUE:
            self.plane = plane.createPlane(100-plane.CENTER[0], 0)
            for i in range(self.team_size):
                player_plane = self.plane.createPlane(400+i*Player.PLAYER_SIZE*2, 0)
                p = Player(player_buffer.__len__()+i, self.team_id, player_plane, (Player.PLAYER_SIZE,)*5, Player.PLAYER_MAX_SPEED)
                self.players.append(p)
                player_buffer.append(p)
        elif self.team_id == TEAM_GREEN:
            self.plane = plane.createPlane(plane.CENTER[0]-100, 0)
            for i in range(self.team_size):
                player_plane = self.plane.createPlane(-400+i*Player.PLAYER_SIZE*2, 0)
                p = Player(player_buffer.__len__()+i, self.team_id, player_plane, (Player.PLAYER_SIZE,)*5, Player.PLAYER_MAX_SPEED)
                self.players.append(p)
                player_buffer.append(p)

    def show(self):
        self.plane.show()


class Playground(Game):

    TEAM_COUNT = 2
    TEAM_SIZE = 5  # in one team

    BALL_SIZE = 3

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 120
        self.set_window()
        self.set_title(self.title)

        self.teams: list[Team] = []
        self.players: list[Player] = []
        self.ball: Ball = None
        self.bodies: list[Body] = []

        self.current_player = -1

    def setup(self):
        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.create_wall()

        self.teams.append(Team(TEAM_GREEN, self.TEAM_SIZE, self.plane, self.players))
        self.teams.append(Team(TEAM_BLUE, self.TEAM_SIZE, self.plane, self.players))

        for player in self.players:
            self.bodies.append(player)

        self.ball = Ball(0, self.plane.createPlane(0, 0), (self.BALL_SIZE,)*10, drag_coef=0.01)
        # self.bodies.append(self.ball)

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

    def loop(self):
        self.check_ball()

        speed = self.players[0].speed()
        if self.keys[core.K_UP]:
            self.players[0].accelerate(5)
        if self.keys[core.K_DOWN]:
            self.players[0].accelerate(-1)
        if self.keys[core.K_LEFT]:
            self.players[0].rotate(Player.PLAYER_MAX_TURN_RATE/(speed+1))
        if self.keys[core.K_RIGHT]:
            self.players[0].rotate(-Player.PLAYER_MAX_TURN_RATE/(speed+1))

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
                    self.current_player = -1

    def onRender(self):
        self.window.fill((255,)*3)
        self.plane.show()
        self.teams[0].show()
        self.engine.step()
        if self.ball.is_free:
            self.ball.show()
        p1 = self.players[0]
        p1_plane = p1.shape.plane
        unit = p1.velocity.unit(100, vector=False)
        vv1 = p1_plane.createVector(unit[0], unit[1])
        vv2 = p1_plane.createVector(unit[0], unit[1])
        vv1.rotate(Player.PLAYER_MAX_FOV/2/180*np.pi)
        vv2.rotate(-Player.PLAYER_MAX_FOV/2/180*np.pi)
        vv1.show()
        vv2.show()
        for player in self.players[1:]:
            pos = player.shape.plane.CENTER
            pos = p1_plane.to_xy(pos)
            v = p1_plane.createVector(pos[0], pos[1])
            ab = np.arccos(p1.velocity.dot(v) / (p1.velocity.mag() * v.mag())) / np.pi*180
            if ab < Player.PLAYER_MAX_FOV/2:
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

    def create_wall(self):
        y = self.size[1] / 2
        for _ in range(28):
            vec = self.plane.createVector(-self.size[0]/2, y)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            vec = self.plane.createVector(self.size[0]/2, y)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            y -= 40

        x = -self.size[0]/2 + 40
        for _ in range(47):
            vec = self.plane.createVector(x, self.size[1] / 2)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            vec = self.plane.createVector(x, -self.size[1] / 2)
            self.bodies.append(
                StaticRectangleBody(-1,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            x += 40
