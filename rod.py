import warp as wp
import numpy as np

def getRotation_low(direction: wp.vec3, initAxis: wp.vec3 = wp.vec3(0, 0, 1)) -> wp.quatf:
    direction = direction / wp.length(direction)
    if direction == initAxis:
        return wp.quat_from_matrix(wp.identity(3, dtype = wp.float32))
    h = initAxis + direction
    h = h / wp.length(h)
    return wp.quat_from_axis_angle(h, wp.PI)

def getRotation(direction: wp.vec3, initAxis: wp.vec3 = wp.vec3(0, 0, 1)) -> wp.quatf:
    if direction == wp.vec3(0, 0, 1):
        return wp.quat_from_matrix(wp.identity(3, dtype = wp.float32))
    axis1 = wp.normalize(wp.cross(wp.vec3(0, 0, 1), direction))
    axis2 = wp.normalize(wp.cross(direction, axis1))
    rotationMat = wp.mat33(axis1[0], axis2[0], direction[0],
                           axis1[1], axis2[1], direction[1],
                           axis1[2], axis2[2], direction[2])
    return wp.quat_from_matrix(rotationMat)

class Rod:

    def __init__(self, n = 100, radius = 0.1, E = 100, nu = 0.1, m = 0.1,
                 start: wp.vec3 = wp.vec3(0, 0, 0), end: wp.vec3 = wp.vec3(10, 0, 0)):
        wp.init()
        self.n = n
        self.length = wp.length(end - start)
        self.radius = radius
        self.l = self.length / n
        self.m = m / (n + 1)
        self.lm = m / n

        rotQuaternion = getRotation((end - start) / self.length)
        self.A1 = wp.PI * radius * radius
        self.A2 = self.A1
        self.A3 = self.A1
        self.J1 = wp.PI * radius ** 4 / 2
        self.J2 = self.J1
        self.J3 = self.J1 * 2

        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

        self.x = wp.from_numpy(
            np.linspace([start[0], start[1], start[2]], [end[0], end[1], end[2]], n + 1, endpoint = True),
            dtype = float, shape = (n + 1, 3)
        ).flatten()
        self.u = wp.from_numpy(
            np.array([rotQuaternion[0], rotQuaternion[1], rotQuaternion[2], rotQuaternion[3]] * n),
            dtype = float, shape = (4 * n, )
        )
        self.v = wp.zeros_like(self.x)
        self.w = wp.zeros(shape = (3 * n, ), dtype = float)

    def debug(self):
        print(self.x)
        print(self.u)
