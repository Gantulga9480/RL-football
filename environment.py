from Game import Game
from Game import core
from Game.graphic import CartesianPlane
from Game.physics import (DynamicPolygonBody,
                          FreePolygonBody, Body,
                          StaticRectangleBody)
from Game.physics import EnginePolygon
import numpy as np


TEAM_COLOR = [(0, 162, 232), (34, 177, 76)]


class Ball(FreePolygonBody):

    def __init__(self,
                 id: int,
                 plane: CartesianPlane,
                 size: tuple,
                 max_speed: float = 0,
                 drag_coef: float = 0) -> None:
        super().__init__(id, plane, size, max_speed, drag_coef)

    def show(self, vertex: bool = False, velocity: bool = False) -> None:
        core.draw.circle(self.shape.plane.window, (255, 0, 0), self.velocity.TAIL, self.radius)
        super().show(vertex, velocity)


class Player(DynamicPolygonBody):

    def __init__(self,
                 id: int,
                 team_id: int,
                 plane: CartesianPlane,
                 size: tuple,
                 max_speed: float = 1,
                 drag_coef: float = 0.03,
                 friction_coef: float = 0.3) -> None:
        super().__init__(id, plane, size, max_speed, drag_coef, friction_coef)
        self.kicked = False
        self.team_id = team_id

    def kick(self, ball: Ball, power: float):
        self.detach(ball)
        d = self.velocity.dir()
        power += 1
        ball.velocity.head = (power*np.cos(d), power*np.sin(d))
        self.kicked = True

    def show(self, vertex: bool = False, velocity: bool = False) -> None:
        core.draw.circle(self.shape.plane.window, TEAM_COLOR[self.team_id], self.velocity.TAIL, self.radius)
        super().show(vertex, velocity)


class Playground(Game):

    TEAM_COUNT = 2
    TEAM_SIZE = 5  # in one team
    PLAYER_SIZE = 10
    BALL_SIZE = 3

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 120
        self.set_window()
        self.set_title(self.title)

        self.players: list[Player] = []
        self.ball: Ball = None
        self.bodies: list[Body] = []

        self.current_player = -1

    def setup(self):
        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        for i in range(self.TEAM_SIZE):
            player_plane = self.plane.createPlane(-400+i*self.PLAYER_SIZE*2, 0)
            p = Player(i+1, 0, player_plane, (self.PLAYER_SIZE,)*5, 10)
            self.bodies.append(p)
            self.players.append(p)

        for i in range(self.TEAM_SIZE):
            player_plane = self.plane.createPlane(400+i*self.PLAYER_SIZE*2, 0)
            p = Player(i+1+self.TEAM_COUNT, 1, player_plane, (self.PLAYER_SIZE,)*5, 10)
            self.bodies.append(p)
            self.players.append(p)

        bplane = self.plane.createPlane(0, 0)
        self.ball = Ball(0, bplane, (self.BALL_SIZE,)*10, drag_coef=0.01)
        ball_1 = Player(1000, 1, bplane.createPlane(-320, 0), (self.PLAYER_SIZE,)*5, 10)
        self.bodies.append(self.ball)
        self.bodies.append(ball_1)

        y = self.size[1] / 2
        for _ in range(28):
            vec = self.plane.createVector(-self.size[0]/2, y)
            self.bodies.append(
                StaticRectangleBody(0,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            vec = self.plane.createVector(self.size[0]/2, y)
            self.bodies.append(
                StaticRectangleBody(0,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            y -= 40

        x = -self.size[0]/2 + 40
        for _ in range(47):
            vec = self.plane.createVector(x, self.size[1] / 2)
            self.bodies.append(
                StaticRectangleBody(0,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            vec = self.plane.createVector(x, -self.size[1] / 2)
            self.bodies.append(
                StaticRectangleBody(0,
                                    CartesianPlane(self.window, (40, 40), vec),
                                    (40, 40)))
            x += 40

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

    def loop(self):
        if self.keys[core.K_UP]:
            self.players[0].accelerate(5)
        if self.keys[core.K_DOWN]:
            self.players[0].accelerate(-2)
        if self.keys[core.K_LEFT]:
            self.players[0].rotate(5)
        if self.keys[core.K_RIGHT]:
            self.players[0].rotate(-5)

        # if self.keys[core.K_f]:
        #     if self.current_player != -1:
        #         self.players[self.current_player].power += 0.1`

        # for player in self.players:
        #     r = np.random.random() * 5 - 2.5
        #     r1 = np.random.random() * 20 - 10
        #     player.accelerate(r1)
        #     player.rotate(r)

        dists = []
        idx = []
        for i, player in enumerate(self.players):
            d = player.shape.plane.get_parent_vector().distance_to(self.ball.shape.plane.get_parent_vector())
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
                if not self.ball.is_attached:
                    player.attach(self.ball, False)
                    self.current_player = p_idx

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
            if event.key == core.K_f:
                if self.current_player != -1:
                    self.players[self.current_player].kick(self.ball, 5)
                    self.current_player = -1

    def onRender(self):
        self.window.fill((255,)*3)
        self.plane.show()
        self.engine.step()
