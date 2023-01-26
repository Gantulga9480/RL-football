import cython
from Game.physics.body cimport DynamicPolygonBody
from Game.graphic.cartesian cimport CartesianPlane
from Game.physics.body cimport Body, STATIC, DYNAMIC
from libc.math cimport cos, sin
import pygame as core


@cython.optimize.unpack_method_calls(False)
cdef class Player(DynamicPolygonBody):

    PLAYER_SIZE = 10
    PLAYER_MAX_SPEED = 3
    PLAYER_SPEED_BALL = 2
    PLAYER_MAX_TURN_RATE = 6
    PLAYER_MAX_FOV = 120

    TEAM_COLOR = [(0, 162, 232), (34, 177, 76)]

    cdef public int team_id
    cdef public bint kicked
    cdef public bint has_ball

    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self,
                 int id,
                 int team_id,
                 CartesianPlane plane,
                 tuple size,
                 double max_speed=1,
                 double drag_coef=0.01,
                 double friction_coef=0.3):
        super().__init__(id, plane, size, max_speed, drag_coef, friction_coef)
        self.team_id = team_id
        self.kicked = False
        self.has_ball = False

    def kick(self, ball, double power):
        cdef double d
        if self.has_ball:
            power += 1
            d = self.velocity.dir()
            ball.velocity.set_head((power * cos(d), power * sin(d)))
            ball.shape.plane.parent_vector.set_head(ball.shape.plane.parent_vector.plane.to_xy(self.shape.plane.get_CENTER()))
            ball.is_free = True
            self.velocity.max = self.PLAYER_MAX_SPEED
            self.kicked = True
            self.has_ball = False

    def reset(self, (double, double) position, double diraction):
        cdef double tmp = diraction - self.velocity.dir()
        self.shape.plane.parent_vector.set_head(position)
        self.velocity.max = self.PLAYER_MAX_SPEED
        self.velocity.rotate(tmp)
        self.shape.rotate(tmp)
        self.kicked = False
        self.has_ball = False

    def show(self, vertex=False, velocity=False):
        core.draw.circle(self.shape.plane.window, self.TEAM_COLOR[self.team_id], self.velocity.TAIL, self.radius)
        super().show(vertex, velocity)
        if self.has_ball:
            core.draw.circle(self.shape.plane.window, (255, 0, 0), self.velocity.TAIL, 3)
            core.draw.circle(self.shape.plane.window, (0, 0, 0), self.velocity.TAIL, 3, 1)
