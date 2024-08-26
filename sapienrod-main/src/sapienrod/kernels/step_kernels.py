import warp as wp
from ..rod_defs import ShapeTypes
from .distance_funcs import edge_egde_closest_point


@wp.kernel
def kinematic_update(
    n_particles: int,
    n_rods: int,
    positions: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.quat),
    linear_velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    particle_masks: wp.array(dtype=float),
    gravity: wp.vec3,
    h: float,
    damping: float,
):
    i = wp.tid()

    if i < n_particles:
        v = (linear_velocities[i] + gravity * h) * particle_masks[i]
        positions[i] = positions[i] + v * (1.0 - damping) * h
        w = angular_velocities[i]

    if i < n_rods:
        q = quaternions[i]
        q = q + (q * wp.quat(w * (1.0 - damping), 0.0)) * (h * 0.5)
        q = wp.normalize(q)
        quaternions[i] = q


@wp.kernel
def compute_velocities(
    n_particles: int,
    n_rods: int,
    positions: wp.array(dtype=wp.vec3),
    positions_prev_step: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.quat),
    quaternions_prev_step: wp.array(dtype=wp.quat),
    linear_velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
    h: float,
):
    i = wp.tid()

    if i < n_particles:
        linear_velocities[i] = (positions[i] - positions_prev_step[i]) / h

    if i < n_rods:
        q_prev = quaternions_prev_step[i]
        q = quaternions[i]
        w_quat = wp.quat_inverse(q_prev) * q * (2.0 / h)
        angular_velocities[i] = wp.vec3(w_quat[0], w_quat[1], w_quat[2])


@wp.kernel
def project_ss_constraints(
    positions: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.quat),
    left_vertex_ids: wp.array(dtype=int),
    inv_masses: wp.array(dtype=float),
    inv_inertias: wp.array(dtype=float),
    rest_basis: wp.array(dtype=wp.vec3, ndim=2),
    lengths: wp.array(dtype=float),
    particle_masks: wp.array(dtype=float),
    ss_stiffness: float,
    h: float,
):
    iq = wp.tid()  # rod id
    i1 = left_vertex_ids[iq]
    i2 = i1 + 1

    p1 = positions[i1]
    p2 = positions[i2]
    q = quaternions[iq]
    l = lengths[iq]
    w1 = inv_masses[i1]
    w2 = inv_masses[i2]
    wq = inv_inertias[iq]

    # e1 = rest_basis[iq, 0]
    # e2 = rest_basis[iq, 1]
    e3 = rest_basis[iq, 2]

    alpha = 1.0 / (ss_stiffness * h * h)

    d3 = wp.quat_rotate(q, e3)
    dLambda_div_l = -l * ((p2 - p1) / l - d3) / (w1 + w2 + l * l * (4.0 * wq + alpha))

    dp1 = -w1 * dLambda_div_l
    dp2 = w2 * dLambda_div_l
    dq_vec = -2.0 * wq * l * dLambda_div_l
    dq = wp.quat(dq_vec, 0.0) * q * wp.quat(-e3, 0.0)

    # positions[i1] = positions[i1] + dp1 * particle_masks[i1]
    # positions[i2] = positions[i2] + dp2 * particle_masks[i2]
    # q_new = q + dq
    # quaternions[iq] = wp.normalize(q_new)
    wp.atomic_add(positions, i1, dp1 * particle_masks[i1])
    wp.atomic_add(positions, i2, dp2 * particle_masks[i2])
    wp.atomic_add(quaternions, iq, dq)
    quaternions[iq] = wp.normalize(quaternions[iq])


@wp.kernel
def project_bt_constraints(
    quaternions: wp.array(dtype=wp.quat),
    left_vertex_ids: wp.array(dtype=int),
    inv_inertias: wp.array(dtype=float),
    lengths: wp.array(dtype=float),
    bt_stiffness: float,
    h: float,
):
    i1 = wp.tid()  # rod id
    i2 = i1 + 1  # rod id
    ip = left_vertex_ids[i2]  # particle id

    q1 = quaternions[i1]
    q2 = quaternions[i2]
    w1 = inv_inertias[i1]
    w2 = inv_inertias[i2]
    l = lengths[i1]

    alpha = 1.0 / (bt_stiffness * h * h)
    alpha *= l * l / 4.0

    omega_rest = wp.quat(0.0, 0.0, 0.0, 1.0)
    omega = wp.quat_inverse(q1) * q2
    d_positive = wp.length_sq(omega - omega_rest)
    d_negative = wp.length_sq(omega + omega_rest)
    sgn = wp.select(d_positive > d_negative, 1.0, -1.0)

    dLambda_bt = -1.0 * (omega - omega_rest * sgn) / (w1 + w2 + alpha)
    dq1 = q2 * wp.quat_inverse(dLambda_bt) * w1
    dq2 = q1 * dLambda_bt * w2

    # quaternions[i1] = wp.normalize(q1 + dq1)
    # quaternions[i2] = wp.normalize(q2 + dq2)
    wp.atomic_add(quaternions, i1, dq1)
    wp.atomic_add(quaternions, i2, dq2)
    quaternions[i1] = wp.normalize(quaternions[i1])
    quaternions[i2] = wp.normalize(quaternions[i2])
    
    
@wp.func
def box_sdf(x: wp.vec3, scale: wp.vec3):
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(x[0]) - scale[0]
    qy = abs(x[1]) - scale[1]
    qz = abs(x[2]) - scale[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def box_sdf_grad(p: wp.vec3, scale: wp.vec3):
    qx = abs(p[0]) - scale[0]
    qy = abs(p[1]) - scale[1]
    qz = abs(p[2]) - scale[2]

    # exterior case
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(p[0], -scale[0], scale[0])
        y = wp.clamp(p[1], -scale[1], scale[1])
        z = wp.clamp(p[2], -scale[2], scale[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if qx > qy and qx > qz or qy == 0.0 and qz == 0.0:
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if qy > qx and qy > qz or qx == 0.0 and qz == 0.0:
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    return wp.vec3(0.0, 0.0, sz)


@wp.kernel
def project_collision_point_body(
    n_shapes: int,
    particle_masks: wp.array(dtype=float),
    positions: wp.array(dtype=wp.vec3),
    positions_prev: wp.array(dtype=wp.vec3),
    particle_radii: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_shape_body_ids: wp.array(dtype=int),
    body_shape_types: wp.array(dtype=int),
    body_shape_scales: wp.array(dtype=wp.vec3),
    body_shape_volumes: wp.array(dtype=wp.uint64),
    body_shape_shape2cm: wp.array(dtype=wp.transform),
    body_shape_fric: wp.array(dtype=float),
    collision_margin: float,
    h: float,
):
    pi = wp.tid()  # particle id
    
    radius = particle_radii[pi]

    px_w = positions[pi]
    px_w_prev = positions_prev[pi]
    px_w_new = px_w

    for si in range(n_shapes):  # shape id
        bi = body_shape_body_ids[si]  # body id

        scale = body_shape_scales[si]
        
        X_wb = body_q[bi]  # body CM to world
        X_wb_prev = body_q_prev[bi]
        X_bs = body_shape_shape2cm[si]  # shape to body CM
        X_ws = wp.transform_multiply(X_wb, X_bs)  # shape to world
        X_ws_prev = wp.transform_multiply(X_wb_prev, X_bs)
        X_sw = wp.transform_inverse(X_ws)
        px_s = wp.transform_point(X_sw, px_w)  # particle x in shape frame
        px_b = wp.transform_point(X_bs, px_s)  # particle x in body frame
        bx_w_prev = wp.transform_point(
            X_wb_prev, px_s
        )  # contact point in world frame at previous step
        bv_w = (px_w - bx_w_prev) / h
        
        d = 1.0e6
        n_s = wp.vec3(0.0, 0.0, 0.0)  # contact normal in shape frame (body -> particle)
        
        shape_type = body_shape_types[si]
        if shape_type == ShapeTypes.GEO_PLANE:
            d = px_s[0]
            n_s = wp.vec3(1.0, 0.0, 0.0)
        elif shape_type == ShapeTypes.GEO_SPHERE:
            d = wp.length(px_s) - scale[0]
            n_s = wp.normalize(px_s)
        elif shape_type == ShapeTypes.GEO_BOX:
            d = box_sdf(px_s, scale)
            n_s = box_sdf_grad(px_s, scale)
        elif shape_type == ShapeTypes.GEO_SDF:
            volume = body_shape_volumes[si]
            px_s_index = wp.volume_world_to_index(volume, wp.cw_div(px_s, scale))
            nn = wp.vec3()
            d = wp.volume_sample_grad_f(volume, px_s_index, wp.Volume.LINEAR, nn)
            d = d * scale[0]
            n_s = wp.normalize(nn)
            
        d = d - radius - collision_margin
        n_w = wp.transform_vector(X_ws, n_s)
        
        if d < 0.0:
            px_w_new = px_w_new - d * n_w * particle_masks[pi]
            
    positions[pi] = px_w_new
            
            
@wp.kernel
def project_collision_rod_rod(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    left_vertex_ids: wp.array(dtype=int),
    inv_masses: wp.array(dtype=float),
    rod_radii: wp.array(dtype=float),
    collision_margin: float,
    h: float,
    hash_grid_radius: float,
):
    iq0 = wp.tid()  # rod id
    i0 = left_vertex_ids[iq0]
    i1 = i0 + 1
    
    x0 = positions[i0]
    x1 = positions[i1]
    r0 = rod_radii[iq0]
    c0 = (x0 + x1) * 0.5
    
    query = wp.hash_grid_query(grid, c0, hash_grid_radius)
    iq1 = int(0)
    contact_cnt = float(0.0)
    sum_dx0 = wp.vec3(0.0, 0.0, 0.0)
    sum_dx1 = wp.vec3(0.0, 0.0, 0.0)
    
    while wp.hash_grid_query_next(query, iq1):
        i2 = left_vertex_ids[iq1]
        i3 = i2 + 1
        skip = iq0 == iq1 or i0 == i2 or i0 == i3 or i1 == i2 or i1 == i3
        
        if not skip:
            x2 = positions[i2]
            x3 = positions[i3]
            r1 = rod_radii[iq1]
            c1 = (x2 + x3) * 0.5
            
            d_thres = r0 + r1 + collision_margin
            # Check bounding sphere distance
            if wp.length(c0 - c1) - wp.length(x0 - c0) - wp.length(x2 - c1) < d_thres:
                t0, t1 = edge_egde_closest_point(x0, x1, x2, x3)
                
                h0 = x0 + (x1 - x0) * t0
                h1 = x2 + (x3 - x2) * t1
                d = wp.length(h0 - h1) - d_thres
                
                if d < 0.0:
                    contact_cnt += 1.0
                    
                    w0 = inv_masses[i0]
                    w1 = inv_masses[i2]
                    
                    n = wp.normalize(h0 - h1)
                    dh0 = -(w0 / (w0 + w1)) * d * n
                    # dh1 = ((w1 / (w0 + w1)) * d * n)
                    
                    sum_dx0 += (1.0 - t0) * dh0
                    sum_dx1 += t0 * dh0
                
    if contact_cnt > 0.0:
        gamma = 0.8
        wp.atomic_add(positions, i0, gamma * sum_dx0)
        wp.atomic_add(positions, i1, gamma * sum_dx0)
        
    # wp.printf("iq0 = %d, Contact: %f\n", iq0, contact_cnt)
            
            
@wp.kernel
def compute_rod_centers(
    positions: wp.array(dtype=wp.vec3),
    left_vertex_ids: wp.array(dtype=int),
    rod_centers: wp.array(dtype=wp.vec3),
):
    iq = wp.tid()
    i0 = left_vertex_ids[iq]
    i1 = i0 + 1
    rod_centers[iq] = (positions[i0] + positions[i1]) * 0.5
    