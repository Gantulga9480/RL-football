import cython
from Game.graphic.cartesian cimport CartesianPlane, Vector2d
from Game.graphic.shapes cimport Polygon, Rectangle, Triangle, Line
from Game.math.core cimport pi, point2d
from pygame.draw import aalines, aaline, circle
from libc.math cimport floor, sqrt
import numpy as np

STATIC = 0
DYNAMIC = 1
FREE = 2

@cython.optimize.unpack_method_calls(False)
cdef class Body:
    def __cinit__(self, *args, **kwargs):
        self.is_attached = False
        self.is_following_dir = False
        self.type = FREE
        self.id = 0
        self.radius = 0
        self.friction_coef = 0
        self.drag_coef = 0
        self.parent_body = None

    def __init__(self, int id, int type):
        self.id = id
        self.type = type

    def show(self, vertex=False, velocity=False):
        self.shape.show(vertex)
        if velocity:
            self.velocity.show((255, 0, 0))

    cpdef void attach(self, Body o, bint follow_dir):
        if not o.is_attached:
            o.is_attached = True
            o.is_following_dir = follow_dir
            o.parent_body = self
            o.shape.plane.parent_vector.set_head_ref(self.shape.plane.parent_vector.get_head_ref())

    cpdef void detach(self, Body o):
        if o.is_attached and o.parent_body is self:
            o.is_attached = False
            o.is_following_dir = False
            o.shape.plane.parent_vector.set_head_ref(point2d(self.shape.plane.parent_vector.get_x(), self.shape.plane.parent_vector.get_y()))

    cpdef void step(self):
        cdef double d
        if self.is_attached and self.is_following_dir:
            d = self.parent_body.velocity.dir() - self.velocity.dir()
            if d != 0:
                self.shape.rotate(d)
                self.velocity.rotate(d)
        else:
            self.USR_step()

    cdef void USR_step(self):
        pass

    cdef void USR_resolve_collision(self, Body o, (double, double) dxy):
        pass

    cdef void USR_resolve_collision_point(self, double dx, double dy):
        pass

    @cython.cdivision(True)
    cpdef void rotate(self, double angle):
        cdef double w
        if not self.is_following_dir:
            w = angle/self.shape.plane.frame_rate
            self.shape.rotate(w)
            self.velocity.rotate(w)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef void scale(self, double factor):
        cdef int i
        cdef double v_len
        cdef double _min = (<Vector2d>self.shape.vertices[0]).mag()
        for i in range(self.shape.vertex_count):
            v_len = (<Vector2d>self.shape.vertices[i]).mag()
            if v_len < _min:
                _min = v_len
        if factor > 1:
            for i in range(self.shape.vertex_count):
                (<Vector2d>self.shape.vertices[i]).scale(factor)
            self.radius *= factor
        elif factor < 1:
            if _min * factor >= self.shape.vertices[0].min:
                for i in range(self.shape.vertex_count):
                    (<Vector2d>self.shape.vertices[i]).scale(factor)
                self.radius *= factor

    cpdef (double, double) position(self):
        return self.shape.plane.get_CENTER()

    @cython.cdivision(True)
    cpdef double direction(self):
        cdef double d = self.velocity.dir()
        if d < 0:
            return (2*pi + d) / pi * 180.0
        else:
            return d / pi * 180.0

    cpdef double speed(self):
        cdef double s = floor((self.velocity.mag() - 1.0) * 1000.0) / 1000.0
        return s if s > 0 else 0.0

@cython.optimize.unpack_method_calls(False)
cdef class FreeBody(Body):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, double max_speed=0, double drag_coef=0):
        if plane.parent_vector is None:
            raise AttributeError("A body can't be made from a base plane, Use child plane instead!")
        super().__init__(id, FREE)
        self.drag_coef = drag_coef
        self.velocity = Vector2d(plane, 1, 0, max_speed, 1)
        self.velocity.rotate(pi/2)

    cdef void USR_step(self):
        cdef double v_len = floor(self.velocity.mag() * 1000.0) / 1000.0
        cdef (double, double) xy
        cdef (double, double) _xy
        if v_len > 1:
            if not self.is_attached:
                xy = self.velocity.get_head()
                _xy = self.velocity.unit_vector(1)
                self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.get_x() + xy[0] - _xy[0],
                                                         self.shape.plane.parent_vector.get_y() + xy[1] - _xy[1]))
            # Drag is applied even if it's attached to another body
            if self.drag_coef:
                self.velocity.add(-(v_len-1) * self.drag_coef)
        else:
            self.velocity.set_head(self.velocity.unit_vector(1))

    @cython.cdivision(True)
    cpdef void accelerate(self, double speed):
        if speed:
            self.velocity.add(speed/self.shape.plane.frame_rate)

@cython.optimize.unpack_method_calls(False)
cdef class StaticBody(Body):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane):
        if plane.parent_vector is None:
            raise AttributeError("A body can't be made from a base plane, Use child plane instead!")
        super().__init__(id, STATIC)
        self.velocity = Vector2d(plane, 1, 0, 1, 1)
        self.velocity.rotate(pi/2)

@cython.optimize.unpack_method_calls(False)
cdef class DynamicBody(Body):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, double max_speed=1, double drag_coef=0.03, double friction_coef=0.3):
        if plane.parent_vector is None:
            raise AttributeError("A body can't be made from a base plane, Use child plane instead!")
        super().__init__(id, DYNAMIC)
        self.drag_coef = drag_coef
        self.friction_coef = friction_coef
        self.velocity = Vector2d(plane, 1, 0, max_speed, 1)
        self.velocity.rotate(pi/2)

    cdef void USR_resolve_collision(self, Body o, (double, double) dxy):
        if o.type == DYNAMIC:
            self.velocity.scale(self.friction_coef)
            o.velocity.scale(self.friction_coef)
            self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.head.x.num + -dxy[0]/2, self.shape.plane.parent_vector.head.y.num + -dxy[1]/2))
            o.shape.plane.parent_vector.set_head((o.shape.plane.parent_vector.head.x.num + dxy[0]/2, o.shape.plane.parent_vector.head.y.num + dxy[1]/2))
        elif o.type == STATIC:
            self.velocity.scale(self.friction_coef)
            self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.head.x.num + -dxy[0], self.shape.plane.parent_vector.head.y.num + -dxy[1]))

    cdef void USR_step(self):
        cdef double v_len = floor(self.velocity.mag() * 1000.0) / 1000.0
        cdef (double, double) xy
        cdef (double, double) _xy
        if v_len > 1:
            if not self.is_attached:
                xy = self.velocity.get_head()
                _xy = self.velocity.unit_vector(1)
                self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.get_x() + xy[0] - _xy[0],
                                                         self.shape.plane.parent_vector.get_y() + xy[1] - _xy[1]))
            # Drag is applied even if it's attached to another body
            if self.drag_coef:
                self.velocity.add(-(v_len-1) * self.drag_coef)
        else:
            self.velocity.set_head(self.velocity.unit_vector(1))

    @cython.cdivision(True)
    cpdef void accelerate(self, double speed):
        if speed:
            self.velocity.add(speed/self.shape.plane.frame_rate)

@cython.optimize.unpack_method_calls(False)
cdef class Ray(FreeBody):

    def __cinit__(self, *args, **kwargs):
        self.x = 0
        self.y = 0

    def __init__(self, int id, CartesianPlane plane, double length, double max_speed=0, double drag_coef=0):
        super().__init__(id, plane, max_speed, drag_coef)
        self.radius = length
        self.shape = Line(plane, length)

    cpdef void reset(self):
        self.x = 0
        self.y = 0

    cdef void USR_resolve_collision_point(self, double dx, double dy):
        if (self.x != 0 or self.y != 0):
            if ((self.x * self.x + self.y * self.y) > (dx * dx + dy * dy)):
                self.x = dx
                self.y = dy
        else:
            self.x = dx
            self.y = dy

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    cpdef void scale(self, double factor):
        (<Vector2d>self.vertices[0]).scale(factor)

    def show(self, vertex=False, velocity=False):
        self.shape.show()
        if self.x != 0 or self.y != 0:
            circle(self.shape.window, (255, 0, 0), self.shape.plane.to_XY((self.x, self.y)), 3)
        self.x = 0
        self.y = 0

@cython.optimize.unpack_method_calls(False)
cdef class DynamicPolygonBody(DynamicBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size, double max_speed=1, double drag_coef=0.03, double friction_coef=0.3):
        super().__init__(id, plane, max_speed, drag_coef, friction_coef)
        self.radius = max(size)
        self.shape = Polygon(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class DynamicRectangleBody(DynamicBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size, double max_speed=1, double drag_coef=0.03, double friction_coef=0.3):
        super().__init__(id, plane, max_speed, drag_coef, friction_coef)
        self.radius = max(size)
        self.shape = Rectangle(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class DynamicTriangleBody(DynamicBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size, double max_speed=1, double drag_coef=0.03, double friction_coef=0.3):
        super().__init__(id, plane, max_speed, drag_coef, friction_coef)
        self.radius = max(size)
        self.shape = Triangle(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class StaticPolygonBody(StaticBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size):
        super().__init__(id, plane)
        self.radius = max(size)
        self.shape = Polygon(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class StaticRectangleBody(StaticBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size):
        super().__init__(id, plane)
        self.radius = max(size)
        self.shape = Rectangle(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class StaticTriangleBody(StaticBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size):
        super().__init__(id, plane)
        self.radius = max(size)
        self.shape = Triangle(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class FreePolygonBody(FreeBody):
    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self, int id, CartesianPlane plane, tuple size, double max_speed=0, double drag_coef=0):
        super().__init__(id, plane, max_speed, drag_coef)
        self.radius = max(size)
        self.shape = Polygon(plane, size)

@cython.optimize.unpack_method_calls(False)
cdef class Ball(FreePolygonBody):

    def __cinit__(self, *args, **kwargs):
        pass

    def __init__(self,
                 int id,
                 CartesianPlane plane,
                 tuple size,
                 double max_speed=0,
                 double drag_coef=0):
        super().__init__(id, plane, size, max_speed, drag_coef)
        self.is_free = True
        self.is_out = False

    cdef void USR_step(self):
        cdef double v_len = floor(self.velocity.mag() * 1000.0) / 1000.0
        cdef (double, double) xy
        cdef (double, double) _xy
        if v_len > 1 and self.is_free:
            if not self.is_attached:
                xy = self.velocity.get_head()
                _xy = self.velocity.unit_vector(1)
                self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.get_x() + xy[0] - _xy[0],
                                                         self.shape.plane.parent_vector.get_y() + xy[1] - _xy[1]))
            # Drag is applied even if it's attached to another body
            if self.drag_coef:
                self.velocity.add(-(v_len-1) * self.drag_coef)
        else:
            if self.is_free:
                self.velocity.set_head(self.velocity.unit_vector(1))
        if (self.shape.plane.parent_vector.get_x() < self.shape.plane.parent_vector.plane.x_min) or \
            (self.shape.plane.parent_vector.get_x() > self.shape.plane.parent_vector.plane.x_max) or \
             (self.shape.plane.parent_vector.get_y() < self.shape.plane.parent_vector.plane.y_min) or \
              (self.shape.plane.parent_vector.get_y() > self.shape.plane.parent_vector.plane.y_max):
              self.is_out = True
        else:
            self.is_out = False

    def reset(self, (double, double) pos):
        self.is_free = True
        self.velocity.set_head_ref(point2d(1, 0))
        self.shape.plane.parent_vector.set_head_ref(point2d(1, 0))
        self.shape.plane.parent_vector.set_head(pos)

    def show(self, vertex=False, velocity=False) -> None:
        circle(self.shape.plane.window, (255, 0, 0), self.velocity.get_TAIL(), self.radius)
        super().show(vertex, velocity)

@cython.optimize.unpack_method_calls(False)
cdef class Player(DynamicPolygonBody):

    def __cinit__(self, *args, **kwargs):
        self.PLAYER_SIZE = 30
        self.PLAYER_MAX_SPEED = 10
        self.PLAYER_SPEED_BALL = 7
        self.PLAYER_MAX_TURN_RATE = 6
        self.PLAYER_MAX_FOV = 120

    def __init__(self,
                 int id,
                 int team_id,
                 CartesianPlane plane,
                 double ability_point = 0.95):
        super().__init__(id, plane.createPlane(0, 0), (self.PLAYER_SIZE,) * 5, self.PLAYER_SPEED_BALL, 0.01, 0.3)
        self.team_id = team_id
        self.kicked = False
        self.has_ball = False
        self.ability_point = ability_point

    def kick(self, FreePolygonBody ball, double power):
        cdef double d
        if self.has_ball:
            power += 1
            ball.is_free = True
            ball.velocity.set_head_ref(point2d(1, 0))
            ball.velocity.set_head(self.velocity.unit_vector(power))
            ball.shape.plane.parent_vector.set_head_ref(point2d(1, 0))
            ball.shape.plane.parent_vector.set_head(self.shape.plane.parent_vector.get_head())

            self.velocity.max = self.PLAYER_MAX_SPEED
            self.kicked = True
            self.has_ball = False

    def reset(self, (double, double) position, double diraction):
        cdef double tmp = diraction - self.velocity.dir()
        self.shape.plane.parent_vector.set_head(position)
        self.velocity.max = self.PLAYER_MAX_SPEED
        self.velocity.rotate(tmp)
        self.shape.rotate(tmp)
        self.velocity.set_head(self.velocity.unit_vector(1))
        self.kicked = False
        self.has_ball = False

    def show(self, color=(150, 150, 150), vertex=False, velocity=False):
        circle(self.shape.plane.window, color, self.velocity.get_TAIL(), self.radius)
        super().show(vertex, velocity)
        if self.has_ball:
            circle(self.shape.plane.window, (255, 0, 0), self.velocity.get_TAIL(), 10)
            circle(self.shape.plane.window, (0, 0, 0), self.velocity.get_TAIL(), 10, 2)

    cdef void USR_resolve_collision(self, Body o, (double, double) dxy):
        if o.type == DYNAMIC:
            self.velocity.scale(self.friction_coef)
            o.velocity.scale(self.friction_coef)
            self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.head.x.num + -dxy[0]/2, self.shape.plane.parent_vector.head.y.num + -dxy[1]/2))
            o.shape.plane.parent_vector.set_head((o.shape.plane.parent_vector.head.x.num + dxy[0]/2, o.shape.plane.parent_vector.head.y.num + dxy[1]/2))
            if self.team_id != o.team_id:
                if np.random.random() > self.ability_point:
                    if self.has_ball:
                        self.has_ball = False
                        self.velocity.max = self.PLAYER_MAX_SPEED
                        o.has_ball = True
                        o.velocity.max = self.PLAYER_SPEED_BALL
                    elif o.has_ball:
                        self.has_ball = True
                        self.velocity.max = self.PLAYER_SPEED_BALL
                        o.has_ball = False
                        o.velocity.max = self.PLAYER_MAX_SPEED
        elif o.type == STATIC:
            self.velocity.scale(self.friction_coef)
            self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.head.x.num + -dxy[0], self.shape.plane.parent_vector.head.y.num + -dxy[1]))


@cython.optimize.unpack_method_calls(False)
cdef class GoalKeeper(DynamicPolygonBody):

    def __cinit__(self, *args, **kwargs):
        self.PLAYER_SIZE = 30
        self.PLAYER_MAX_SPEED = 10
        self.PLAYER_SPEED_BALL = 10
        self.PLAYER_MAX_TURN_RATE = 10
        self.PLAYER_MAX_FOV = 120

    def __init__(self,
                 int id,
                 int team_id,
                 CartesianPlane plane,
                 double ability_point = 1):
        super().__init__(id, plane.createPlane(0, 0), (self.PLAYER_SIZE,) * 5, self.PLAYER_SPEED_BALL, 0.01, 0.3)
        self.team_id = team_id
        self.kicked = False
        self.has_ball = False
        self.ability_point = ability_point
        self.shape.plane.parent_vector.max = 200

    def kick(self, FreePolygonBody ball, double power):
        cdef double d
        if self.has_ball:
            power += 1
            ball.velocity.set_head(self.velocity.unit_vector(power))
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
        self.velocity.set_head(self.velocity.unit_vector(1))
        self.kicked = False
        self.has_ball = False

    def show(self, color=(150, 150, 150), vertex=False, velocity=False):
        circle(self.shape.plane.window, color, self.velocity.get_TAIL(), self.radius)
        super().show(vertex, velocity)
        if self.has_ball:
            circle(self.shape.plane.window, (255, 0, 0), self.velocity.get_TAIL(), 10)
            circle(self.shape.plane.window, (0, 0, 0), self.velocity.get_TAIL(), 10, 2)

    cdef void USR_resolve_collision(self, Body o, (double, double) dxy):
        if o.type == DYNAMIC:
            self.velocity.scale(self.friction_coef)
            o.velocity.scale(self.friction_coef)
            self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.head.x.num + -dxy[0]/2, self.shape.plane.parent_vector.head.y.num + -dxy[1]/2))
            o.shape.plane.parent_vector.set_head((o.shape.plane.parent_vector.head.x.num + dxy[0]/2, o.shape.plane.parent_vector.head.y.num + dxy[1]/2))
            if self.team_id != o.team_id:
                if np.random.random() > self.ability_point:
                    if self.has_ball:
                        self.has_ball = False
                        self.velocity.max = self.PLAYER_MAX_SPEED
                        o.has_ball = True
                        o.velocity.max = self.PLAYER_SPEED_BALL
                    elif o.has_ball:
                        self.has_ball = True
                        self.velocity.max = self.PLAYER_SPEED_BALL
                        o.has_ball = False
                        o.velocity.max = self.PLAYER_MAX_SPEED
        elif o.type == STATIC:
            self.velocity.scale(self.friction_coef)
            self.shape.plane.parent_vector.set_head((self.shape.plane.parent_vector.head.x.num + -dxy[0], self.shape.plane.parent_vector.head.y.num + -dxy[1]))
