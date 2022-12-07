from Game.graphic.cartesian cimport CartesianPlane
from Game.physics.body cimport object_body


cdef class collision:

    cdef CartesianPlane plane

    cpdef void check(self, object_body b1, object_body b2)
    cdef void diagonal_intersect(self, object_body body1, object_body body2)
