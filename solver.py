import warp as wp
import numpy as np
from dbc import *
from geometry import *

@wp.kernel
def queryDiagonal(transIndex: wp.array(dtype = int),
                  rotIndex: wp.array(dtype = int),
                  collisionCount: wp.array(dtype = int),
                  transtmin: wp.array(dtype = float),
                  transtmax: wp.array(dtype = float),
                  rottmin: wp.array(dtype = float),
                  rottmax: wp.array(dtype = float),
                  M_star: wp.array(dtype = float),
                  res: wp.array(dtype = float), # stores the result
                  n: int, transNum: int, rotNum: int,
                  wSE: float, wBT: float, lInv: float, currentTime: float):
    j, i = wp.tid()
    if j == 0 and i < n: # SE potential
        base3 = 3 * i
        base4 = 3 * n + 3 + 4 * i
        wlInv = wSE * lInv
        wlInv_square = wlInv * lInv
        wp.atomic_add(res, base3, wlInv_square)
        wp.atomic_add(res, base3 + 1, wlInv_square)
        wp.atomic_add(res, base3 + 2, wlInv_square)
        wp.atomic_add(res, base3 + 3, wlInv_square)
        wp.atomic_add(res, base3 + 4, wlInv_square)
        wp.atomic_add(res, base3 + 5, wlInv_square)
        wp.atomic_add(res, base4, wSE)
        wp.atomic_add(res, base4 + 1, wSE)
        wp.atomic_add(res, base4 + 2, wSE)
        wp.atomic_add(res, base4 + 3, wSE)
    elif j == 1 and i < n - 1: # BT potential
        base3 = 3 * i
        base4 = 3 * n + 3 + 4 * i
        wp.atomic_add(res, base4, wBT)
        wp.atomic_add(res, base4 + 1, wBT)
        wp.atomic_add(res, base4 + 2, wBT)
        wp.atomic_add(res, base4 + 3, wBT)
        wp.atomic_add(res, base4 + 4, wBT)
        wp.atomic_add(res, base4 + 5, wBT)
        wp.atomic_add(res, base4 + 6, wBT)
        wp.atomic_add(res, base4 + 7, wBT)
    elif j == 2: # DBC potential
        if i >= transNum and i < transNum + rotNum:
            index = i - transNum
            if currentTime <= rottmax[index] and currentTime >= rottmin[index]:
                base4 = 4 * rotIndex[index]
                # iBase4 = 4 * index
                start = 3 * n + 3
                wp.atomic_add(res, base4 + start, DBCWeight)
                wp.atomic_add(res, base4 + start + 1, DBCWeight)
                wp.atomic_add(res, base4 + start + 2, DBCWeight)
                wp.atomic_add(res, base4 + start + 3, DBCWeight)
        elif i < transNum:
            index = i
            if currentTime <= transtmax[index] and currentTime >= transtmin[index]:
                base3 = 3 * transIndex[index]
                # iBase3 = 3 * i
                wp.atomic_add(res, base3, DBCWeight)
                wp.atomic_add(res, base3 + 1, DBCWeight)
                wp.atomic_add(res, base3 + 2, DBCWeight)
    elif j == 3 and i <= n: # Collision potential
        base3 = 3 * i
        count = wp.float32(collisionCount[i])
        wp.atomic_add(res, base3, CollisionWeight * count)
        wp.atomic_add(res, base3 + 1, CollisionWeight * count)
        wp.atomic_add(res, base3 + 2, CollisionWeight * count)
    elif j == 4: # M_star
        wp.atomic_add(res, i, M_star[i])
    # elif j == 5: # self-collision
    #     pass

@wp.kernel
def precondition(diagonal: wp.array(dtype = float),
                 vec: wp.array(dtype = float)):
    i = wp.tid()
    vec[i] /= diagonal[i]

@wp.kernel
def queryProduct(transIndex: wp.array(dtype = int),
                 rotIndex: wp.array(dtype = int),
                 collisionCount: wp.array(dtype = int),
                 transtmin: wp.array(dtype = float),
                 transtmax: wp.array(dtype = float),
                 rottmin: wp.array(dtype = float),
                 rottmax: wp.array(dtype = float),
                 M_star: wp.array(dtype = float),
                 vec: wp.array(dtype = float), # the vector to multiply
                 res: wp.array(dtype = float), # stores the result
                 n: int, transNum: int, rotNum: int,
                 wSE: float, wBT: float, lInv: float, currentTime: float):
    j, i = wp.tid()
    if j == 0 and i < n: # SE potential
        base3 = 3 * i
        base4 = 3 * n + 3 + 4 * i
        wlInv = wSE * lInv
        wlInv_square = wlInv * lInv
        wp.atomic_add(res, base3, vec[base3] * wlInv_square)
        wp.atomic_add(res, base3 + 1, vec[base3 + 1] * wlInv_square)
        wp.atomic_add(res, base3 + 2, vec[base3 + 2] * wlInv_square)
        wp.atomic_add(res, base3 + 3, vec[base3 + 3] * wlInv_square)
        wp.atomic_add(res, base3 + 4, vec[base3 + 4] * wlInv_square)
        wp.atomic_add(res, base3 + 5, vec[base3 + 5] * wlInv_square)
        wp.atomic_add(res, base3, vec[base3 + 3] * -wlInv_square)
        wp.atomic_add(res, base3 + 1, vec[base3 + 4] * -wlInv_square)
        wp.atomic_add(res, base3 + 2, vec[base3 + 5] * -wlInv_square)
        wp.atomic_add(res, base3 + 3, vec[base3] * -wlInv_square)
        wp.atomic_add(res, base3 + 4, vec[base3 + 1] * -wlInv_square)
        wp.atomic_add(res, base3 + 5, vec[base3 + 2] * -wlInv_square)
        wp.atomic_add(res, base4, vec[base4] * wSE)
        wp.atomic_add(res, base4 + 1, vec[base4 + 1] * wSE)
        wp.atomic_add(res, base4 + 2, vec[base4 + 2] * wSE)
        wp.atomic_add(res, base4 + 3, vec[base4 + 3] * wSE)
    elif j == 1 and i < n - 1: # BT potential
        base3 = 3 * i
        base4 = 3 * n + 3 + 4 * i
        wp.atomic_add(res, base4, vec[base4] * wBT)
        wp.atomic_add(res, base4 + 1, vec[base4 + 1] * wBT)
        wp.atomic_add(res, base4 + 2, vec[base4 + 2] * wBT)
        wp.atomic_add(res, base4 + 3, vec[base4 + 3] * wBT)
        wp.atomic_add(res, base4 + 4, vec[base4 + 4] * wBT)
        wp.atomic_add(res, base4 + 5, vec[base4 + 5] * wBT)
        wp.atomic_add(res, base4 + 6, vec[base4 + 6] * wBT)
        wp.atomic_add(res, base4 + 7, vec[base4 + 7] * wBT)
    elif j == 2: # DBC potential
        if i >= transNum and i < transNum + rotNum:
            index = i - transNum
            if currentTime <= rottmax[index] and currentTime >= rottmin[index]:
                base4 = 4 * rotIndex[index]
                # iBase4 = 4 * index
                start = 3 * n + 3
                wp.atomic_add(res, base4 + start, vec[base4 + start] * DBCWeight)
                wp.atomic_add(res, base4 + start + 1, vec[base4 + start + 1] * DBCWeight)
                wp.atomic_add(res, base4 + start + 2, vec[base4 + start + 2] * DBCWeight)
                wp.atomic_add(res, base4 + start + 3, vec[base4 + start + 3] * DBCWeight)
        elif i < transNum:
            index = i
            if currentTime <= transtmax[index] and currentTime >= transtmin[index]:
                base3 = 3 * transIndex[index]
                # iBase3 = 3 * i
                wp.atomic_add(res, base3, vec[base3] * DBCWeight)
                wp.atomic_add(res, base3 + 1, vec[base3 + 1] * DBCWeight)
                wp.atomic_add(res, base3 + 2, vec[base3 + 2] * DBCWeight)
    elif j == 3 and i <= n: # Collision potential
        base3 = 3 * i
        count = wp.float32(collisionCount[i])
        wp.atomic_add(res, base3, vec[base3] * CollisionWeight * count)
        wp.atomic_add(res, base3 + 1, vec[base3 + 1] * CollisionWeight * count)
        wp.atomic_add(res, base3 + 2, vec[base3 + 2] * CollisionWeight * count)
    elif j == 4: # M_star
        wp.atomic_add(res, i, vec[i] * M_star[i])


@wp.kernel
def dot(a: wp.array(dtype = float),
        b: wp.array(dtype = float),
        res: wp.array(dtype = float),
        stride: int, n: int):
    i = wp.tid()
    tmp = float(0.0)
    for k in range(i, n, stride):
        tmp += a[k] * b[k]
    wp.atomic_add(res, 0, tmp)

@wp.kernel
def vectorAdd(x: wp.array(dtype = float),
              y: wp.array(dtype = float),
              res: wp.array(dtype = float),
              alpha: wp.array(dtype = float), 
              beta: wp.array(dtype = float),
              scale: float):
    i = wp.tid()
    res[i] = x[i] + (alpha[0] / beta[0]) * y[i] * scale
