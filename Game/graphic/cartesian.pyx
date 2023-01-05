import cython
from pygame.draw import line, aaline
from Game.math.core cimport scalar, point2d, vector2d
from libc.math cimport floor, sqrt
from random import random


@cython.optimize.unpack_method_calls(False)
cdef class CartesianPlane:

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, window, (double, double) window_size, Vector2d parent_vector=None, double frame_rate=60, double unit_length=1):
        if unit_length <= 0:
            raise ValueError("Wrong unit length")
        self.window = window
        self.window_size = window_size
        self.parent_vector = parent_vector
        self.frame_rate = frame_rate
        self.unit_length = (unit_length if self.parent_vector is None
                            else self.parent_vector.plane.unit_length)
        self.center = (self.parent_vector.headXY if self.parent_vector
                       else point2d(floor(self.window_size[0] / 2), floor(self.window_size[1] / 2)))
        self.set_limit()

    @property
    def X(self):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center.x.num

    @property
    def Y(self):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center.y.num

    @property
    def CENTER(self):
        if self.parent_vector:
            self.parent_vector.update()
        return (self.center.x.num, self.center.y.num)

    @property
    def SIZE(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def createVector(self, x=1, y=0, max=0, min=0):
        return Vector2d(self, x, y, max, min)

    def createRandomVector(self, max=0, min=0):
        vec = Vector2d(self, 1, 0, max, min)
        vec.random()
        return vec

    def createPlane(self, x=0, y=0):
        return CartesianPlane(self.window, self.window_size, Vector2d(self, x, y), self.frame_rate)

    @cython.optimize.unpack_method_calls(False)
    def show(self):
        if self.parent_vector:
            self.parent_vector.update()
        # draw x axis
        line(self.window, (255, 0, 0), self.center.get_xy(),
             (self.center.x.num, self.center.y.num-10), 2)
        # draw y axis
        line(self.window, (0, 255, 0), self.center.get_xy(),
             (self.center.x.num+10, self.center.y.num), 2)

    cpdef Vector2d get_parent_vector(self):
        return self.parent_vector

    cpdef point2d get_center_point(self):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center

    cpdef double to_X(self, double x):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center.x.num + x * self.unit_length

    cpdef double to_Y(self, double y):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center.y.num - y * self.unit_length

    cpdef (double, double) to_XY(self, (double, double) xy):
        if self.parent_vector:
            self.parent_vector.update()
        return (self.center.x.num + xy[0] * self.unit_length, self.center.y.num - xy[1] * self.unit_length)

    @cython.cdivision(True)
    cpdef double to_x(self, double X):
        if self.parent_vector:
            self.parent_vector.update()
        return (X - self.center.x.num) / self.unit_length

    @cython.cdivision(True)
    cpdef double to_y(self, double Y):
        if self.parent_vector:
            self.parent_vector.update()
        return (self.center.y.num - Y) / self.unit_length

    @cython.cdivision(True)
    cpdef (double, double) to_xy(self, (double, double) XY):
        if self.parent_vector:
            self.parent_vector.update()
        return ((XY[0] - self.center.x.num) / self.unit_length, (self.center.y.num - XY[1]) / self.unit_length)

    cdef double get_X(self):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center.x.num

    cdef double get_Y(self):
        if self.parent_vector:
            self.parent_vector.update()
        return self.center.y.num

    cdef (double, double) get_CENTER(self):
        if self.parent_vector:
            self.parent_vector.update()
        return (self.center.x.num, self.center.y.num)

    @cython.cdivision(True)
    cdef void set_limit(self):
        if self.parent_vector:
            self.parent_vector.update()
        self.x_min = floor((self.center.x.num - self.window_size[0]) / self.unit_length)
        self.x_max = floor((self.window_size[0] - self.center.x.num) / self.unit_length)
        self.y_min = floor((self.center.y.num - self.window_size[1]) / self.unit_length)
        self.y_max = floor((self.window_size[1] - self.center.y.num) / self.unit_length)


@cython.optimize.unpack_method_calls(False)
cdef class Vector2d(vector2d):

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, CartesianPlane plane, double x=1, double y=0, double max=0, double min=0):
        self.plane = plane
        self.headXY = point2d(0, 0)
        super().__init__(x, y, max, min)
        self.update()

    def __repr__(self):
        return f'X:{self.X}, Y:{self.Y} - x:{self.head.x}, y:{self.head.y}'

    @property
    def X(self):
        self.update()
        return self.headXY.x.num

    @property
    def Y(self):
        self.update()
        return self.headXY.y.num

    @property
    def HEAD(self):
        self.update()
        return self.headXY.get_xy()

    @property
    def TAIL(self):
        self.update()
        return (self.plane.center.x.num, self.plane.center.y.num)

    @cython.nonecheck(False)
    @cython.optimize.unpack_method_calls(False)
    def show(self, color=(0, 0, 0)):
        self.update()
        aaline(self.plane.window, color, self.plane.center.get_xy(), self.headXY.get_xy())

    def unit(self, double scale=1, bint vector=True):
        cdef (double, double) xy = self.unit_vector(scale)
        if vector:
            return Vector2d(self.plane, xy[0], xy[1], self.max, self.min)
        return xy

    def normal(self, double scale=1, bint vector=True):
        cdef (double, double) xy = self.normal_vector(scale)
        if vector:
            return Vector2d(self.plane, xy[0], xy[1], self.max, self.min)
        return xy

    cpdef double dist(self, Vector2d vector):
        self.update()
        vector.update()
        cdef double dx = self.headXY.x.num - vector.headXY.x.num
        cdef double dy = self.headXY.y.num - vector.headXY.y.num
        return sqrt(dx*dx + dy*dy)

    cpdef void random(self):
        # TODO Fix null vector creation
        cdef double r1 = random()
        cdef double r2 = random()
        if self.max:
            self.head.x.num = (r1 * 2 - 1) * self.max
            self.head.y.num = (r2 * 2 - 1) * self.max
        else:
            self.head.x.num = (r1 * 2 - 1) * self.plane.x_max
            self.head.y.num = (r2 * 2 - 1) * self.plane.y_max

    cpdef void update(self):
        self.headXY.set_xy(self.plane.to_XY(self.head.get_xy()))

    cdef double get_X(self):
        self.update()
        return self.headXY.x.num

    cdef double get_Y(self):
        self.update()
        return self.headXY.y.num

    cdef (double, double) get_HEAD(self):
        self.update()
        return self.headXY.get_xy()

    cdef (double, double) get_TAIL(self):
        self.update()
        return (self.plane.center.x.num, self.plane.center.y.num)
