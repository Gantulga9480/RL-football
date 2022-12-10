import cython
from libc.math cimport cos, sin, atan2, sqrt
from libc.math cimport abs as cabs

pi = 3.141592653589793

@cython.optimize.unpack_method_calls(False)
@cython.nonecheck(False)
cdef class scalar:

    def __cinit__(self, *args, **kwargs):
        self.num = 0
        self.min = 0
        self.max = 0

    def __init__(self, double value, (double, double) limits=(0, 0)):
        if limits[0] or limits[1]:
            self.min = limits[0]
            self.max = limits[1]
            if value < self.min:
                self.num = self.min
            elif value > self.max:
                self.num = self.max
            else:
                self.num = value
        else:
            self.num = value

    @property
    def value(self):
        return self.num

    @value.setter
    def value(self, double o):
        if self.max:
            if (self.min <= o) and (o <= self.max):
                self.num = o
            else:
                self.num = self.min if o < self.min else self.max
        else:
            self.num = o

    def __repr__(self):
        return str(self.num)

    cdef void add(self, double o):
        self.set_value(self.num + o)

    cdef void scale(self, double o):
        self.set_value(self.num * o)

    cdef double get_value(self):
        return self.num

    cdef void set_value(self, double o):
        if self.max:
            if self.min <= o <= self.max:
                self.num = o
            else:
                self.num = self.min if o < self.min else self.max
        else:
            self.num = o


@cython.optimize.unpack_method_calls(False)
@cython.nonecheck(False)
cdef class point2d:

    def __cinit__(self, *args, **kwargs):
        ...

    def __init__(self, double x, double y, (double, double) x_lim=(0, 0), (double, double) y_lim=(0, 0)):
        self.x = scalar(x, x_lim)
        self.y = scalar(y, y_lim)

    @property
    def x(self):
        return self.x.num

    @x.setter
    def x(self, double o):
        self.x.set_value(o)

    @property
    def y(self):
        return self.y.num

    @y.setter
    def y(self, double o):
        self.y.set_value(o)

    @property
    def xy(self):
        return (self.x.num, self.y.num)

    @xy.setter
    def xy(self, (double, double) o):
        self.x.set_value(o[0])
        self.y.set_value(o[1])

    cpdef void set_x_ref(self, scalar o):
        self.x = o

    cpdef void set_y_ref(self, scalar o):
        self.y = o

    cpdef scalar get_x_ref(self):
        return self.x

    cpdef scalar get_y_ref(self):
        return self.y

    cdef void set_x(self, double o):
        self.x.set_value(o)

    cdef double get_x(self):
        return self.x.num

    cdef void set_y(self, double o):
        self.y.set_value(o)

    cdef double get_y(self):
        return self.y.num

    cdef (double, double) get_xy(self):
        return (self.x.num, self.y.num)

    cdef void set_xy(self, (double, double) o):
        self.x.set_value(o[0])
        self.y.set_value(o[1])


@cython.optimize.unpack_method_calls(False)
@cython.nonecheck(False)
cdef class vector2d:

    def __cinit__(self, *args, **kwargs):
        self.max = 0
        self.min = 0

    def __init__(self, double x, double y, double max = 0, double min = 0):
        self.head = point2d(x, y)
        self.max = max
        self.min = min

    @property
    def x(self):
        return self.head.x.num

    @x.setter
    def x(self, double o):
        cdef double _len = sqrt(o * o + self.head.y.num * self.head.y.num)
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head.x.set_value(o)
            else:
                self.head.x.set_value(o)

    @property
    def y(self):
        return self.head.y.num

    @y.setter
    def y(self, double o):
        cdef double _len = sqrt(o * o + self.head.x.num * self.head.x.num)
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head.y.set_value(o)
            else:
                self.head.y.set_value(o)

    @property
    def head(self):
        return (self.head.x.num, self.head.y.num)

    @head.setter
    def head(self, (double, double) o):
        cdef double _len = sqrt(o[0]*o[0] + o[1]*o[1])
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head.x.set_value(o[0])
                    self.head.y.set_value(o[1])
            else:
                self.head.x.set_value(o[0])
                self.head.y.set_value(o[1])

    def unit(self, double scale=1, bint vector=True):
        cdef double a = self.dir()
        cdef double x = cos(a) * scale
        cdef double y = sin(a) * scale
        if vector:
            return vector2d(x, y, self.max, self.min)
        return (x, y)

    def normal(self, double scale=1, bint vector=True):
        cdef double x = -self.head.y.num * scale
        cdef double y = self.head.x.num * scale
        if vector:
            return vector2d(x, y, self.max, self.min)
        return (x, y)

    cpdef void set_x_ref(self, scalar o):
        #TODO
        raise NotImplementedError

    cpdef void set_y_ref(self, scalar o):
        #TODO
        raise NotImplementedError

    cpdef void set_head_ref(self, point2d o):
        cdef double x = o.x.num
        cdef double y = o.y.num
        cdef double _len = sqrt(x*x + y*y)
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head = o
            else:
                self.head = o

    cpdef scalar get_x_ref(self):
        return self.head.get_x_ref()

    cpdef scalar get_y_ref(self):
        return self.head.get_y_ref()

    cpdef point2d get_head_ref(self):
        return self.head

    cpdef void add(self, double o):
        cdef double a
        cdef (double, double) xy
        if o > 0:
            if self.max:
                if (self.mag() + o) <= self.max:
                    a = self.dir()
                    self.head.x.add(o * cos(a))
                    self.head.y.add(o * sin(a))
                else:
                    xy = self.unit_vector(self.max)
                    self.head.x.num = xy[0]
                    self.head.y.num = xy[1]
            else:
                a = self.dir()
                self.head.x.add(o * cos(a))
                self.head.y.add(o * sin(a))
        elif o < 0:
            if cabs(o) < (self.mag() - self.min):
                a = self.dir()
                self.head.x.add(o * cos(a))
                self.head.y.add(o * sin(a))
            else:
                xy = self.unit_vector(self.min)
                self.head.x.num = xy[0]
                self.head.y.num = xy[1]

    cpdef void scale(self, double o):
        cdef (double, double) xy
        cdef double v_len = self.mag()
        if o > 1:
            if self.max:
                if (v_len * o) < self.max:
                    self.head.x.scale(o)
                    self.head.y.scale(o)
                else:
                    xy = self.unit_vector(self.max)
                    self.head.x.num = xy[0]
                    self.head.y.num = xy[1]
            else:
                self.head.x.scale(o)
                self.head.y.scale(o)
        elif 0 <= o < 1:
            if (v_len * o) > self.min:
                self.head.x.scale(o)
                self.head.y.scale(o)
            else:
                xy = self.unit_vector(self.min)
                self.head.x.num = xy[0]
                self.head.y.num = xy[1]

    cpdef void rotate(self, double radians):
        cdef double x = self.head.x.num
        cdef double y = self.head.y.num
        self.head.x.set_value(x * cos(radians) - y * sin(radians))
        self.head.y.set_value(x * sin(radians) + y * cos(radians))

    cpdef double mag(self):
        return sqrt(self.head.x.num*self.head.x.num + self.head.y.num*self.head.y.num)

    cpdef double dir(self):
        return atan2(self.head.y.num, self.head.x.num)

    cpdef double distance_to(self, vector2d vector):
        cdef double dx = self.head.x.num - vector.head.x.num
        cdef double dy = self.head.y.num - vector.head.y.num
        return sqrt(dx*dx + dy*dy)

    cpdef double angle_between(self, vector2d vector):
        return atan2(vector.head.y.num, vector.head.x.num) - self.dir()

    cpdef double dot(self, vector2d vector):
        cdef double x = -self.head.y.num
        cdef double y = self.head.x.num
        cdef double _x = -vector.head.y.num
        cdef double _y = vector.head.x.num
        return x * _x + y * _y

    cdef void set_x(self, double o):
        cdef double _len = sqrt(o * o + self.head.y.num * self.head.y.num)
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head.x.set_value(o)
            else:
                self.head.x.set_value(o)

    cdef void set_y(self, double o):
        cdef double _len = sqrt(o * o + self.head.x.num * self.head.x.num)
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head.y.set_value(o)
            else:
                self.head.y.set_value(o)

    cdef void set_head(self, (double, double) o):
        cdef double _len = sqrt(o[0]*o[0] + o[1]*o[1])
        if _len >= self.min:
            if self.max:
                if _len <= self.max:
                    self.head.x.set_value(o[0])
                    self.head.y.set_value(o[1])
            else:
                self.head.x.set_value(o[0])
                self.head.y.set_value(o[1])

    cdef double get_x(self):
        return self.head.x.num

    cdef double get_y(self):
        return self.head.y.num

    cdef (double, double) get_head(self):
        return (self.head.x.num, self.head.y.num)

    cdef (double, double) unit_vector(self, double scale):
        cdef double a = self.dir()
        cdef double x = cos(a) * scale
        cdef double y = sin(a) * scale
        return (x, y)

    cdef (double, double) normal_vector(self, double scale):
        cdef double x = -self.head.y.num * scale
        cdef double y = self.head.x.num * scale
        return (x, y)
