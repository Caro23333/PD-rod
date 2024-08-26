import warp as wp


@wp.kernel
def update_rod_meshes(
    rods_begin: int,
    positions: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.quat),
    rest_basis: wp.array(dtype=wp.vec3, ndim=2),
    left_vertex_ids: wp.array(dtype=int),
    mesh_V: wp.array(dtype=wp.vec3),
    mesh_V_begin: int,
    n_V: int,
    radius: float,
    render_V: wp.array(dtype=float, ndim=2),
):
    i = wp.tid() + rods_begin  # rod id
    p0 = positions[left_vertex_ids[i]]
    p1 = positions[left_vertex_ids[i] + 1]
    p = (p0 + p1) / 2.0
    q = quaternions[i]
    l = wp.length(p1 - p0)
    
    e1 = rest_basis[i, 0]
    e2 = rest_basis[i, 1]
    e3 = rest_basis[i, 2]
    
    for j in range(n_V):
        v = mesh_V[mesh_V_begin + j]
        v = wp.cw_mul(v, wp.vec3(radius, radius, l / 2.0))
        v = v[0] * e1 + v[1] * e2 + v[2] * e3
        v = wp.quat_rotate(q, v) + p
        render_V[i * n_V + j, 0] = v[0]
        render_V[i * n_V + j, 1] = v[1]
        render_V[i * n_V + j, 2] = v[2]
        

@wp.kernel
def update_particle_meshes(
    particle_begin: int,
    positions: wp.array(dtype=wp.vec3),
    mesh_V: wp.array(dtype=wp.vec3),
    mesh_V_begin: int,
    n_V: int,
    radius: float,
    render_V_begin: int,
    render_V: wp.array(dtype=float, ndim=2),
):
    i = wp.tid()
    x = positions[i + particle_begin]
    
    for j in range(n_V):
        v = mesh_V[j + mesh_V_begin] * radius * 1.5
        v = v + x
        render_V[i * n_V + j + render_V_begin, 0] = v[0]
        render_V[i * n_V + j + render_V_begin, 1] = v[1]
        render_V[i * n_V + j + render_V_begin, 2] = v[2]

    
    
