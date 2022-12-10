from Game.graphic.cartesian cimport CartesianPlane
from Game.physics.body cimport Body


cdef class collision:

    cdef CartesianPlane plane

    cpdef void check(self, Body b1, Body b2)
    cdef void diagonal_intersect(self, Body body1, Body body2)
