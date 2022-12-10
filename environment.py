from Game import Game
from Game import core
from Game.graphic import CartesianPlane
from Game.physics import (DynamicPolygonBody,
                          FreePolygonBody, Body,
                          StaticRectangleBody)
from Game.physics import EnginePolygon
import numpy as np


class Ball(FreePolygonBody):

    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 0, drag_coef: float = 0) -> None:
        super().__init__(id, plane, size, max_speed, drag_coef)


class Player(DynamicPolygonBody):

    def __init__(self, id: int, plane: CartesianPlane, size: tuple, max_speed: float = 1, drag_coef: float = 0.03, friction_coef: float = 0.3) -> None:
        super().__init__(id, plane, size, max_speed, drag_coef, friction_coef)
        self.kicked = False
        self.power = 0


class Playground(Game):

    PLAYER_COUNT = 10
    PLAYER_SIZE = 20
    BALL_SIZE = 5

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 60
        self.set_window()
        self.set_title(self.title)

        self.p_plane: list[CartesianPlane] = []
        self.b_plane: CartesianPlane = None
        self.players: list[Player] = []
        self.ball: Ball = None
        self.bodies: list[Body] = []

        self.current_player = -1

    def setup(self):
        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.b_plane = self.plane.createPlane(0, 400)
        self.ball = Ball(0, self.b_plane, (self.BALL_SIZE,)*10, 20, drag_coef=0.01)
        self.bodies.append(self.ball)
        for i in range(self.PLAYER_COUNT):
            pp = self.plane.createPlane(-400+i*self.PLAYER_SIZE*2, 0)
            self.p_plane.append(pp)
            p = Player(i+1, pp, (self.PLAYER_SIZE,)*5, 10)
            self.bodies.append(p)
            self.players.append(p)

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

        if self.keys[core.K_f]:
            if self.current_player != -1:
                self.players[self.current_player].power += 0.1

        # for player in self.players:
        #     r = np.random.random() * 5 - 2.5
        #     r1 = np.random.random() * 20 - 10
        #     player.accelerate(r1)
        #     player.rotate(r)

        for i, player in enumerate(self.players):
            d = player.shape.plane.get_parent_vector().distance_to(self.ball.shape.plane.get_parent_vector())
            if (d <= (player.radius + self.ball.radius)):
                if player.kicked:
                    pass
                else:
                    if not self.ball.is_attached:
                        player.attach(self.ball, False)
                        self.current_player = i
            else:
                player.kicked = False

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_f:
                if self.current_player != -1:
                    p = self.players[self.current_player]
                    p.detach(self.ball)
                    d = p.velocity.dir()
                    self.ball.velocity.head = (p.power*np.cos(d), p.power*np.sin(d))
                    p.kicked = True
                    p.power = 0
                    self.current_player = -1

    def onRender(self):
        self.window.fill((255,)*3)
        self.plane.show()
        self.engine.step()
