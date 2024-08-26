import numpy as np
import warp as wp
import warp.sparse as sparse

wp.init()
DBCWeight = wp.constant(10.0 ** 6)

@wp.kernel
def advanceTransDBC(x: wp.array(dtype = float),
                    v: wp.array(dtype = float),
                    tmin: wp.array(dtype = float),
                    currentTime: float, dt: float):
    i = wp.tid()
    if currentTime >= tmin[i]:
        t = wp.min(currentTime - tmin[i], dt)
        base = 3 * i
        x[base] += v[base] * t
        x[base + 1] += v[base + 1] * t
        x[base + 2] += v[base + 2] * t

@wp.kernel
def advanceRotDBC(u: wp.array(dtype = float),
                  axis: wp.array(dtype = wp.vec3),
                  w: wp.array(dtype = float),
                  tmin: wp.array(dtype = float),
                  currentTime: float, dt: float):
    i = wp.tid()
    if currentTime >= tmin[i]:
        t = wp.min(currentTime - tmin[i], dt)
        base = 4 * i
        u0 = wp.quaternion(u[base], u[base + 1], u[base + 2], u[base + 3])
        q = wp.quat_from_axis_angle(axis[i], w[i] * t)
        u1 = q * u0
        u[base] = u1[0]
        u[base + 1] = u1[1]
        u[base + 2] = u1[2]
        u[base + 3] = u1[3]

@wp.kernel
def DBCPotential(x: wp.array(dtype = float),
                 u: wp.array(dtype = float),
                 q: wp.array(dtype = float),
                 transtmin: wp.array(dtype = float),
                 transtmax: wp.array(dtype = float),
                 rottmin: wp.array(dtype = float),
                 rottmax: wp.array(dtype = float),
                 transIndex: wp.array(dtype = int),
                 rotIndex: wp.array(dtype = int),
                 b: wp.array(dtype = float),
                 n: int, transNum: int, currentTime: float):
    i = wp.tid()
    if i >= transNum:
        index = i - transNum
        if currentTime <= rottmax[index] and currentTime >= rottmin[index]:
            base4 = 4 * rotIndex[index]
            iBase4 = 4 * index
            start = 3 * n + 3
            wp.atomic_add(b, base4 + start, DBCWeight * u[iBase4])
            wp.atomic_add(b, base4 + start + 1, DBCWeight * u[iBase4 + 1])
            wp.atomic_add(b, base4 + start + 2, DBCWeight * u[iBase4 + 2])
            wp.atomic_add(b, base4 + start + 3, DBCWeight * u[iBase4 + 3])
    else:
        index = i
        if currentTime <= transtmax[index] and currentTime >= transtmin[index]:
            base3 = 3 * transIndex[index]
            iBase3 = 3 * i
            wp.atomic_add(b, base3, DBCWeight * x[iBase3])
            wp.atomic_add(b, base3 + 1, DBCWeight * x[iBase3 + 1])
            wp.atomic_add(b, base3 + 2, DBCWeight * x[iBase3 + 2])