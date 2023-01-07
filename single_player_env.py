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

# Actions
KICK = 0
GO_FORWARD = 1
STOP = 2
TURN_LEFT = 3
TURN_RIGHT = 4
ACTIONS = [KICK, GO_FORWARD, STOP, TURN_LEFT, TURN_RIGHT]


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
    PLAYER_MAX_TURN_RATE = 5
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


class SinglePlayer(Game):

    TEAM_SIZE = 1  # in one team
    BALL_SIZE = 3

    def __init__(self) -> None:
        super().__init__()
        self.size = (1920, 1080)
        self.fps = 120
        self.set_window()
        self.set_title(self.title)
        self.done = False

        self.teams: list[Team] = []
        self.players: list[Player] = []
        self.ball: Ball = None
        self.current_player = -1
        self.last_player = -1

        self.bodies: list[Body] = []

        self.counter = 0

    def setup(self):
        self.plane = CartesianPlane(self.window, self.size, frame_rate=self.fps)
        self.create_wall()

        self.teams.append(None)
        self.teams.append(Team(TEAM_RIGHT, self.TEAM_SIZE, self.plane, self.players))

        for player in self.players:
            self.bodies.append(player)

        self.engine = EnginePolygon(self.plane, np.array(self.bodies, dtype=Body))

        self.ball = Ball(0, self.plane.createPlane(650, 0), (self.BALL_SIZE,) * 10, drag_coef=0.01)

    def step(self, action):
        dir_last = self.players[0].direction()
        if action == KICK:
            self.players[0].kick(self.ball, 5)
        elif action == GO_FORWARD:
            self.players[0].accelerate(5)
        elif action == STOP:
            self.players[0].accelerate(-1)
        elif action == TURN_LEFT:
            speed = self.players[0].speed()
            self.players[0].rotate(Player.PLAYER_MAX_TURN_RATE / (speed + 1))
        elif action == TURN_RIGHT:
            speed = self.players[0].speed()
            self.players[0].rotate(-Player.PLAYER_MAX_TURN_RATE / (speed + 1))
        self.loop_once()
        dir_now = self.players[0].direction()
        omega = dir_now - dir_last
        self.counter += 1
        if self.counter == (self.fps * 10):
            self.done = True
        if self.teams[TEAM_RIGHT].score:
            self.done = True
        ball_pos = self.ball.position()
        player_pos = self.plane.to_xy(self.players[0].shape.plane.CENTER)
        speed = self.players[0].speed()
        state = [ball_pos[0], ball_pos[1], player_pos[0], player_pos[1], dir_now, speed, omega]
        if self.done and self.teams[TEAM_RIGHT].score == 0:
            reward = -1
        else:
            reward = self.teams[TEAM_RIGHT].score
        return state, reward, self.done

    def reset(self):
        self.counter = 0
        self.teams[TEAM_RIGHT].score = 0
        self.done = False
        self.ball.velocity.head = (1, 0)
        self.ball.shape.plane.get_parent_vector().head = (650, 0)
        x_lim = self.teams[TEAM_RIGHT].plane.to_xy(self.plane.CENTER)[0]
        y_lim = 450
        x = np.random.randint(x_lim, 1)
        y = np.random.randint(-y_lim, y_lim + 1)
        self.players[0].shape.plane.get_parent_vector().head = (x, y)
        dr = np.random.random() * np.pi * 2
        tmp = dr - self.players[0].velocity.dir()
        self.players[0].velocity.rotate(tmp)
        self.players[0].shape.rotate(tmp)
        ball_pos = self.ball.position()
        dir_now = self.players[0].direction()
        player_pos = self.plane.to_xy(self.players[0].shape.plane.CENTER)
        state = [ball_pos[0], ball_pos[1], player_pos[0], player_pos[1], dir_now, 0, 0]
        return state

    def loop(self):
        self.check_ball()

    def onEvent(self, event):
        if event.type == core.KEYUP:
            if event.key == core.K_q:
                self.running = False
                self.done = True

    def onRender(self):
        self.window.fill((255, 255, 255))
        self.draw_field()
        self.plane.show()
        self.teams[TEAM_RIGHT].show()
        self.engine.step()
        if self.ball.is_free:
            self.ball.show()
        self.players[0].velocity.show()

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
                    # self.ball.velocity.head = (1, 0)
                    # self.ball.shape.plane.get_parent_vector().head = (0, 0)
                    # self.teams[TEAM_LEFT].score += 1
                    pass
                # Right team scored a goal
                elif (pos[0] > self.plane.x_max - 60 and -70 < pos[1] < 70):
                    # self.ball.velocity.head = (1, 0)
                    # self.ball.shape.plane.get_parent_vector().head = (0, 0)
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
