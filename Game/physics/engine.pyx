import cython
from Game.graphic.cartesian cimport CartesianPlane
from Game.physics.body cimport Body, STATIC, FREE
from Game.physics.collision cimport collision

@cython.optimize.unpack_method_calls(False)
cdef class EnginePolygon:

    cdef Body[:] bodies
    cdef collision collider

    def __init__(self, CartesianPlane plane, Body[:] bodies):
        self.bodies = bodies
        self.collider = collision(plane)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.initializedcheck(False)
    def step(self):
        cdef int n = <int>self.bodies.shape[0]
        cdef int i, j
        cdef (double, double) dxy
        # Check every body ...
        for i in range(n):
            # Take one gentle step in environment
            (<Body>self.bodies[i]).step()
            # Will not check STATIC body
            if self.bodies[i].type == STATIC:
                continue
            # ... against every other body
            for j in range(n):
                # Will not check bodies against itself
                # Will not check against FREE body
                # Bodies that have same id will be skipped
                if i == j or self.bodies[j].type == FREE or self.bodies[i].id == self.bodies[j].id:
                    continue
                # radius1 + radius2 >= distance between body2 and body1 means we have some work to do
                if (self.bodies[i].radius + self.bodies[j].radius) >= ((<Body>self.bodies[i]).shape.plane.parent_vector.dist((<Body>self.bodies[j]).shape.plane.parent_vector)):
                    self.collider.check(<Body>self.bodies[i], <Body>self.bodies[j])
