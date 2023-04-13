cdef double pi

cdef class scalar:
    cdef public double min, max
    cdef double num

    cdef void add(self, double o)
    cdef void scale(self, double o)
    cdef double get_value(self)
    cdef void set_value(self, double o)

cdef class point2d:
    cdef scalar x, y

    cpdef void set_x_ref(self, scalar o)
    cpdef void set_y_ref(self, scalar o)
    cpdef scalar get_x_ref(self)
    cpdef scalar get_y_ref(self)
    cdef void set_x(self, double o)
    cdef void set_y(self, double o)
    cdef void set_xy(self, (double, double) o)
    cdef double get_x(self)
    cdef double get_y(self)
    cdef (double, double) get_xy(self)

cdef class vector2d:
    cdef point2d head
    cdef public double max, min

    cpdef void add(self, double o)
    cpdef void scale(self, double o)
    cpdef void set_x_ref(self, scalar o)
    cpdef void set_y_ref(self, scalar o)
    cpdef void set_head_ref(self, point2d o)
    cpdef scalar get_x_ref(self)
    cpdef scalar get_y_ref(self)
    cpdef point2d get_head_ref(self)
    cpdef void rotate(self, double radians)
    cpdef double mag(self)
    cpdef double dir(self)
    cpdef double dist(self, vector2d vector)
    cpdef double angle_between(self, vector2d vector)
    cpdef double dot(self, vector2d vector)
    cdef void set_x(self, double o)
    cdef void set_y(self, double o)
    cdef void set_head(self, (double, double) o)
    cdef double get_x(self)
    cdef double get_y(self)
    cdef (double, double) get_head(self)
    cdef (double, double) unit_vector(self, double scale)
    cdef (double, double) normal_vector(self, double scale)
