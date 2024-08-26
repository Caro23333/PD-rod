import warp as wp
import numpy as np
import warp.sparse as sparse

wp.init()
@wp.kernel
def potential(q: wp.array(dtype = float),
              b: wp.array(dtype = float),
              lInv: float, n: int, wSE: float, wBT: float):
    j, i = wp.tid()
    num = j * n + i
    if j == 1 and i == n - 1:
        return
    base3 = 3 * i
    base4 = 3 * n + 3 + 4 * i
    if j == 0: # SE potential
        wlInv = wSE * lInv
        wlInv_square = wlInv * lInv
        # compute pi first
        xf = wp.vec3(
            lInv * (q[base3 + 3] - q[base3]), 
            lInv * (q[base3 + 4] - q[base3 + 1]), 
            lInv * (q[base3 + 5] - q[base3 + 2])
        )
        un = wp.quaternion(q[base4], q[base4 + 1], q[base4 + 2], q[base4 + 3])
        d3 = wp.normalize(wp.quat_rotate(un, wp.vec3(0.0, 0.0, 1.0)))
        x = wp.cross(d3, xf)
        if wp.length(x) == 0:
            dun = wp.quaternion(0.0, 0.0, 0.0, 1.0)
        else:
            axis = wp.normalize(x)
            theta = wp.acos(wp.dot(xf, d3) / wp.length(xf))
            dun = wp.quat_from_axis_angle(axis, theta)
        un_star = wp.normalize(dun * un) # order remains to check here!!!
        # update right vector
        wp.atomic_add(b, base3, -wlInv * d3[0])
        wp.atomic_add(b, base3 + 1, -wlInv * d3[1])
        wp.atomic_add(b, base3 + 2, -wlInv * d3[2])
        wp.atomic_add(b, base3 + 3, wlInv * d3[0])
        wp.atomic_add(b, base3 + 4, wlInv * d3[1])
        wp.atomic_add(b, base3 + 5, wlInv * d3[2])
        wp.atomic_add(b, base4, wSE * un_star[0])
        wp.atomic_add(b, base4 + 1, wSE * un_star[1])
        wp.atomic_add(b, base4 + 2, wSE * un_star[2])
        wp.atomic_add(b, base4 + 3, wSE * un_star[3])
    else: # BT potential
        # compute pi first
        uNow = wp.quaternion(q[base4], q[base4 + 1], q[base4 + 2], q[base4 + 3])
        uNxt = wp.quaternion(q[base4 + 4], q[base4 + 5], q[base4 + 6], q[base4 + 7])
        uNow_star = wp.quat_slerp(uNow, uNxt, 0.5)
        uNxt_star = uNow_star
        # update right vector
        wp.atomic_add(b, base4, wBT * uNow_star[0])
        wp.atomic_add(b, base4 + 1, wBT * uNow_star[1])
        wp.atomic_add(b, base4 + 2, wBT * uNow_star[2])
        wp.atomic_add(b, base4 + 3, wBT * uNow_star[3])
        wp.atomic_add(b, base4 + 4, wBT * uNxt_star[0])
        wp.atomic_add(b, base4 + 5, wBT * uNxt_star[1])
        wp.atomic_add(b, base4 + 6, wBT * uNxt_star[2])
        wp.atomic_add(b, base4 + 7, wBT * uNxt_star[3])
