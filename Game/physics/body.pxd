import cython
from Game.graphic.cartesian cimport CartesianPlane, Vector2d
from Game.graphic.shapes cimport Shape
from Game.math.core cimport point2d


cdef int STATIC, DYNAMIC, FREE


cdef class object_body:
    cdef public int type, id
    cdef public double radius
    cdef public double friction_factor
    cdef public Shape shape
    cdef public Vector2d velocity
    cdef bint is_attached
    cdef bint is_following_dir
    cdef object_body parent_body

    cpdef void step(self)
    cpdef void rotate(self, double angle)
    cpdef void scale(self, double factor)
    cpdef (double, double) position(self)
    cpdef double direction(self)
    cpdef double speed(self)
    cpdef void attach(self, object_body o, bint follow_dir)
    cpdef void detach(self, object_body o)
    cpdef void USR_step(self)
    cpdef void USR_resolve_collision(self, object_body o, (double, double) dxy)
    cpdef void USR_resolve_collision_point(self, double dx, double dy)

cdef class FreeBody(object_body):
    cpdef void Accelerate(self, double factor)

cdef class StaticBody(object_body):
    pass

cdef class DynamicBody(object_body):
    cdef double max_speed
    cpdef void Accelerate(self, double factor)

cdef class DynamicPolygonBody(DynamicBody):
    pass

cdef class DynamicRectangleBody(DynamicBody):
    pass

cdef class DynamicTriangleBody(DynamicBody):
    pass

cdef class StaticPolygonBody(StaticBody):
    pass

cdef class StaticRectangleBody(StaticBody):
    pass

cdef class StaticTriangleBody(StaticBody):
    pass

cdef class FreePolygonBody(FreeBody):
    pass

cdef class Ray(FreeBody):
    cdef readonly double x
    cdef readonly double y
    cpdef void reset(self)
