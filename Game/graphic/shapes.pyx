import cython
from Game.graphic.cartesian cimport CartesianPlane, Vector2d
from Game.math.core cimport pi
from libc.math cimport sqrt, atan2
import numpy as np
from pygame.draw import aalines, aaline

@cython.optimize.unpack_method_calls(False)
cdef class Shape:

    def __cinit__(self, *args, **kwargs):
        self.vertex_count = 0

    def __init__(self, CartesianPlane plane):
        self.window = plane.window
        self.plane = plane
        self.color = (0, 0, 0)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cpdef void rotate(self, double angle):
        cdef int i
        for i in range(self.vertex_count):
            (<Vector2d>self.vertices[i]).rotate(angle)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cpdef void scale(self, double factor):
        cdef int i
        cdef double v_len
        cdef double _min = (<Vector2d>self.vertices[0]).mag()
        for i in range(self.vertex_count):
            v_len = (<Vector2d>self.vertices[i]).mag()
            if v_len < _min:
                _min = v_len
        if factor > 1:
            for i in range(self.vertex_count):
                (<Vector2d>self.vertices[i]).scale(factor)
        elif factor < 1:
            if _min * factor >= self.vertices[0].min_length:
                for i in range(self.vertex_count):
                    (<Vector2d>self.vertices[i]).scale(factor)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def show(self, show_vertex=False):
        cdef int i
        cdef list heads = []
        if show_vertex:
            for i in range(self.vertex_count):
                (<Vector2d>self.vertices[i]).show(self.color)
                heads.append((<Vector2d>self.vertices[i]).headXY.get_xy())
        else:
            for i in range(self.vertex_count):
                heads.append((<Vector2d>self.vertices[i]).get_HEAD())
        aalines(self.window, self.color, True, heads)

@cython.optimize.unpack_method_calls(False)
cdef class Line(Shape):

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, CartesianPlane plane, double length):
        super().__init__(plane)

        self.vertex_count = 1

        cdef list vers = []
        cdef int i

        for i in range(self.vertex_count):
            vers.append(Vector2d(self.plane, length, 0, 0, 1))

        self.vertices = np.array(vers, dtype=Vector2d)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cpdef void rotate(self, double angle):
        (<Vector2d>self.vertices[0]).rotate(angle)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    cpdef void scale(self, double factor):
        (<Vector2d>self.vertices[0]).scale(factor)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def show(self, show_vertex=False):
        (<Vector2d>self.vertices[0]).update()
        aaline(self.window, self.color, self.plane.center.get_xy(), (<Vector2d>self.vertices[0]).headXY.get_xy())

@cython.optimize.unpack_method_calls(False)
cdef class Rectangle(Shape):

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, CartesianPlane plane, (double, double) shape):
        super().__init__(plane)

        self.vertex_count = 4

        cdef double length = sqrt((shape[0]//2)**2 + (shape[1]//2)**2)

        cdef list vers = []
        cdef int i
        cdef double angle[4]
        cdef double a1 = atan2(shape[1], shape[0])
        cdef double a2 = pi - a1
        angle[:] = [a1, a2, -a2, -a1]

        for i in range(self.vertex_count):
            vers.append(Vector2d(self.plane, length, 0, 0, 1))
            vers[-1].rotate(angle[i])

        self.vertices = np.array(vers, dtype=Vector2d)

@cython.optimize.unpack_method_calls(False)
cdef class Triangle(Shape):

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, CartesianPlane plane, (double, double, double) shape):
        super().__init__(plane)

        self.vertex_count = 3

        cdef list vers = []
        cdef int i

        for i in range(self.vertex_count):
            vers.append(Vector2d(self.plane, shape[i], 0, 0, 1))
            vers[-1].rotate(pi/2 + 2*pi/3 * i)

        self.vertices = np.array(vers, dtype=Vector2d)

@cython.optimize.unpack_method_calls(False)
cdef class Polygon(Shape):

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, CartesianPlane plane, tuple shape):
        super(Polygon, self).__init__(plane)

        self.vertex_count = shape.__len__()

        cdef list vers = []
        cdef int i

        for i in range(self.vertex_count):
            # TODO ?
            vers.append(Vector2d(self.plane, shape[i], 0, 0, 1))
            vers[-1].rotate(pi/2 + 2*pi/self.vertex_count * i)

        self.vertices = np.array(vers, dtype=Vector2d)
