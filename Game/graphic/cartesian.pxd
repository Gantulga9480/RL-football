from Game.math.core cimport point2d, vector2d


cdef class CartesianPlane:
    cdef Vector2d parent_vector
    cdef point2d center
    cdef readonly object window
    cdef readonly (double, double) window_size
    cdef readonly double x_min, y_min, x_max, y_max, unit_length, frame_rate

    cpdef Vector2d get_parent_vector(self)
    cpdef point2d get_center_point(self)
    cpdef double to_X(self, double x)
    cpdef double to_Y(self, double y)
    cpdef (double, double) to_XY(self, (double, double) xy)
    cpdef double to_x(self, double X)
    cpdef double to_y(self, double Y)
    cpdef (double, double) to_xy(self, (double, double) XY)
    cdef double get_X(self)
    cdef double get_Y(self)
    cdef (double, double) get_CENTER(self)
    cdef void set_limit(self)


cdef class Vector2d(vector2d):

    cdef readonly CartesianPlane plane
    cdef point2d headXY

    cpdef double dist(self, Vector2d vector)
    cpdef void random(self)
    cpdef void update(self)
    cdef double get_X(self)
    cdef double get_Y(self)
    cdef (double, double) get_HEAD(self)
    cdef (double, double) get_TAIL(self)
