import cython

@cython.cdivision(True)
cpdef double LSI(double p0x, double p0y, double p1x, double p1y, double p2x, double p2y, double p3x, double p3y):
    cdef double s10_x = p1x - p0x
    cdef double s10_y = p1y - p0y
    cdef double s32_x = p3x - p2x
    cdef double s32_y = p3y - p2y
    cdef double denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0:
        return 0  # collinear
    cdef double denom_is_positive = denom > 0
    cdef double s02_x = p0x - p2x
    cdef double s02_y = p0y - p2y
    cdef double s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denom_is_positive:
        return 0  # no collision
    cdef double t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive:
        return 0  # no collision
    if (s_numer > denom) == denom_is_positive or \
            (t_numer > denom) == denom_is_positive:
        return 0  # no collision
    # collision detected
    return t_numer / denom
    # intersection_point = [p0[0] + (t * s10_x), p0[1] + (t * s10_y)]