import warp as wp


@wp.func
def edge_egde_closest_point(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    e0 = x1 - x0
    e1 = x3 - x2
    x02 = x2 - x0

    A = wp.dot(e0, e0)
    B = -wp.dot(e0, e1)
    C = wp.dot(e1, e1)

    D = wp.dot(e0, x02)
    E = -wp.dot(e1, x02)

    D0 = A * C - B * B
    N0 = C * D - B * E

    no_EE = wp.length_sq(wp.cross(e0, e1)) < 1e-6 * A * C

    if N0 <= 0.0 or (no_EE and N0 < D0 * 0.5):
        N1 = E
        D1 = C
        t0 = 0.0
    elif N0 >= D0 or (no_EE and N0 >= D0 * 0.5):
        N1 = E - B
        D1 = C
        t0 = 1.0
    else:
        N1 = A * E - B * D
        D1 = D0
        t0 = N0 / D0
    
    if N1 <= 0.0:
        t1 = 0.0
        if D <= 0.0:
            t0 = 0.0
        elif D >= A:
            t0 = 1.0
        else:
            t0 = D / A
    elif N1 >= D1:
        t1 = 1.0
        if D - B <= 0.0:
            t0 = 0.0
        elif D - B >= A:
            t0 = 1.0
        else:
            t0 = (D - B) / A
    else:
        t1 = N1 / D1

    return t0, t1
