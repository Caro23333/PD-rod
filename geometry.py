import warp as wp
import numpy as np
import warp.sparse as sparse
from distance import *

# wp.set_device("cpu")
@wp.struct
class SignedDistanceField:
    dummy: int

@wp.struct
class Sphere:

    pos: wp.vec3
    velocity: wp.vec3
    radius: float

@wp.struct
class Table:

    p0: wp.vec3
    p1: wp.vec3
    p2: wp.vec3
    p3: wp.vec3
    norm0: wp.vec3
    norm1: wp.vec3
    norm2: wp.vec3
    norm3: wp.vec3
    norm4: wp.vec3

def initNewSphere(pos, velocity, radius) -> Sphere:
    sphere = Sphere()
    sphere.pos = pos
    sphere.velocity = velocity
    sphere.radius = radius
    return sphere

def initNewTable(p0, p1, p2, p3, reverse) -> Table:
    table = Table()
    table.p0 = p0
    table.p1 = p1
    table.p2 = p2
    table.p3 = p3
    table.norm0 = wp.normalize(wp.cross(p2 - p0, p1 - p0))
    table.norm1 = wp.normalize(wp.cross(table.norm0, p1 - p0))
    table.norm2 = wp.normalize(wp.cross(table.norm0, p2 - p1))
    table.norm3 = wp.normalize(wp.cross(table.norm0, p3 - p2))
    table.norm4 = wp.normalize(wp.cross(table.norm0, p0 - p3))
    if reverse:
        table.norm0 = -1.0 * table.norm0
    return table

@wp.func
def SDFEvaluate(obj: Sphere, pos: wp.vec3, time: float):
    objCurrentPos = obj.pos + time * obj.velocity
    return wp.length(pos - objCurrentPos) - obj.radius

@wp.func
def SDFGradient(obj: Sphere, pos: wp.vec3, time: float):
    objCurrentPos = obj.pos + time * obj.velocity
    return wp.normalize(pos - objCurrentPos)


CollisionWeight = wp.constant(4 * 10.0 ** 4)
rodCollisionWeight = wp.constant(4 * 10.0 ** 5)
ratio = wp.constant(1 * 10 ** 1)
threshold = wp.constant(1 * 10.0 ** -3)
damping = wp.constant(0.8)

@wp.kernel
def sphereCollisionPotential(q: wp.array(dtype = float),
                             spheres: wp.array(dtype = Sphere),
                             collisionCount: wp.array(dtype = int),
                             b: wp.array(dtype = float),
                             rodRadius: float, time: float):
    i, j = wp.tid()
    base3 = 3 * i
    point = wp.vec3(q[base3], q[base3 + 1], q[base3 + 2])
    sphere = spheres[j]
    if SDFEvaluate(sphere, point, time) < rodRadius + threshold:
        grad = SDFGradient(sphere, point, time)
        center = sphere.pos + time * sphere.velocity
        dest = center + (sphere.radius + rodRadius + threshold * 2.0) * grad
        wp.atomic_add(collisionCount, i, 1)
        wp.atomic_add(b, base3, CollisionWeight * dest[0])
        wp.atomic_add(b, base3 + 1, CollisionWeight * dest[1])
        wp.atomic_add(b, base3 + 2, CollisionWeight * dest[2])

@wp.kernel
def tableCollisionPotential(q: wp.array(dtype = float),
                            tables: wp.array(dtype = Table),
                            collisionCount: wp.array(dtype = int),
                            b: wp.array(dtype = float),
                            rodRadius: float):
    i, j = wp.tid()
    base3 = 3 * i
    point = wp.vec3(q[base3], q[base3 + 1], q[base3 + 2])
    table = tables[j]
    hasCollision = False
    dest = wp.vec3(0.0, 0.0, 0.0)
    SDF = -10.0 ** 10.0
    v = point - table.p0  
    d = wp.dot(table.norm0, v)  
    if d > SDF:
        SDF = d
        dest = point + (-d + rodRadius + threshold) * damping * table.norm0 
    v = point - table.p0  
    d = wp.dot(table.norm1, v)  
    if d > SDF:
        SDF = d
        dest = point + (-d + rodRadius + threshold) * damping * table.norm1 
    v = point - table.p1  
    d = wp.dot(table.norm2, v)  
    if d > SDF:
        SDF = d
        dest = point + (-d + rodRadius + threshold) * damping * table.norm2 
    v = point - table.p2  
    d = wp.dot(table.norm3, v)  
    if d > SDF:
        SDF = d
        dest = point + (-d + rodRadius + threshold) * damping * table.norm3 
    v = point - table.p3  
    d = wp.dot(table.norm4, v)  
    if d > SDF:
        SDF = d
        dest = point + (-d + rodRadius + threshold) * damping * table.norm4
    if SDF < rodRadius + threshold:
        wp.atomic_add(collisionCount, i, 1)
        wp.atomic_add(b, base3, CollisionWeight * dest[0])
        wp.atomic_add(b, base3 + 1, CollisionWeight * dest[1])
        wp.atomic_add(b, base3 + 2, CollisionWeight * dest[2])


@wp.kernel
def rodCollisionPotential(q: wp.array(dtype = float),
                          collisionCount: wp.array(dtype = int), 
                          b: wp.array(dtype = float),
                          n: int, rodRadius: float, 
                          grid: wp.uint64, hash_grid_radius: float):
    i0 = wp.tid()
    i1 = i0 + 1
    base3 = 3 * i0
    x0 = wp.vec3(q[base3], q[base3 + 1], q[base3 + 2])
    x1 = wp.vec3(q[base3 + 3], q[base3 + 4], q[base3 + 5])
    c0 = (x0 + x1) * 0.5

    query = wp.hash_grid_query(grid, c0, hash_grid_radius)
    iq1 = int(0)
    
    while wp.hash_grid_query_next(query, iq1):
        i2 = iq1
        i3 = i2 + 1
        skip = i0 == i2 or i0 == i3 or i1 == i2 or i1 == i3

        if not skip:
            qbase3 = 3 * i2
            x2 = wp.vec3(q[qbase3], q[qbase3 + 1], q[qbase3 + 2])
            x3 = wp.vec3(q[qbase3 + 3], q[qbase3 + 4], q[qbase3 + 5])
            c1 = (x2 + x3) * 0.5
            d_thres = 2.0 * rodRadius + threshold
            if True:
                t0, t1 = edge_egde_closest_point(x0, x1, x2, x3)
                h0 = x0 + (x1 - x0) * t0
                h1 = x2 + (x3 - x2) * t1
                d = wp.length(h0 - h1) - d_thres

                if d < 0.0 and ((t0 != 0.0 and t0 != 1.0) or (t1 != 0.0 and t1 != 1.0)):
                    wp.atomic_add(collisionCount, i0, ratio)
                    wp.atomic_add(collisionCount, i1, ratio)
                    norm = wp.normalize(h0 - h1)
                    d *= damping
                    wp.atomic_add(b, base3, rodCollisionWeight * (q[base3] + norm[0] * 0.5 * -d))
                    wp.atomic_add(b, base3 + 1, rodCollisionWeight * (q[base3 + 1] + norm[1] * 0.5 * -d))
                    wp.atomic_add(b, base3 + 2, rodCollisionWeight * (q[base3 + 2] + norm[2] * 0.5 * -d))
                    wp.atomic_add(b, base3 + 3, rodCollisionWeight * (q[base3 + 3] + norm[0] * 0.5 * -d))
                    wp.atomic_add(b, base3 + 4, rodCollisionWeight * (q[base3 + 4] + norm[1] * 0.5 * -d))
                    wp.atomic_add(b, base3 + 5, rodCollisionWeight * (q[base3 + 5] + norm[2] * 0.5 * -d))