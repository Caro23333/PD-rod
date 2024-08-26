import warp as wp
import numpy as np
import weakref
import sapien

from .rod_component import *
from .utils.array import wp_slice
from .kernels.step_kernels import *


class RodConfig:
    def __init__(self):
        ################ Memory config ################
        self.max_scenes = 1 << 10
        self.max_bodies = 1 << 10
        self.max_components = 1 << 10
        self.max_particles = 1 << 20

        ################ Solver config ################
        self.time_step = 1e-3
        self.n_pbd_iters = 10
        self.hash_grid_dim = 128

        ################ Physics config ################
        self.gravity = np.array([0, 0, -9.8], dtype=np.float32)
        self.stretch_shear_stiffness = 1e3
        self.bend_twist_stiffness = 2e-5  # TODO: replace with component stiffness
        self.collision_margin = 1e-3
        self.collision_weight = 5e3
        self.max_velocity = (
            1.0  # estimated max velocity for collision detection, TODO: add clamping
        )
        self.damping = 2e-3  # damping energy: 0.5 * damping * v^2


class RodSystem(sapien.System):
    def __init__(
        self,
        config: RodConfig = None,
        device="cuda:0",
    ):
        super().__init__()

        wp.init()

        self.name = "rod"

        self.config = config
        self.device = device

        self.scenes = []
        self.components = []

        self._init_arrays()
        self._init_counters()

    def get_name(self):
        return self.name

    def _init_arrays(self):
        config = self.config
        MB = config.max_bodies
        MP = config.max_particles

        with wp.ScopedDevice(self.device):
            # ================== Body data ==================
            self.body_q = wp.zeros(MP, dtype=wp.transform)
            self.body_q_prev = wp.zeros(MP, dtype=wp.transform)
            self.body_qd = wp.zeros(MP, dtype=wp.spatial_vector)
            self.body_f_ext = wp.zeros(MP, dtype=wp.spatial_vector)
            self.body_shape_types = wp.zeros(MP, dtype=int)
            self.body_shape_scales = wp.zeros(MP, dtype=wp.vec3)
            self.body_shape_fric = wp.zeros(MP, dtype=float)
            self.body_shape_volumes = wp.zeros(MP, dtype=wp.uint64)
            self.body_shape_shape2cm = wp.zeros(MP, dtype=wp.transform)
            self.body_shape_body_ids = wp.zeros(MP, dtype=int)

            # ================== Rod data ==================
            self.positions = wp.zeros(MP, dtype=wp.vec3)
            self.positions_prev_step = wp.zeros(MP, dtype=wp.vec3)
            self.positions_rest = wp.zeros(MP, dtype=wp.vec3)
            self.rod_centers = wp.zeros(MP, dtype=wp.vec3)
            self.quaternions = wp.zeros(MP, dtype=wp.quat)  # [x, y, z, w]
            self.quaternions_prev_step = wp.zeros(MP, dtype=wp.quat)
            self.linear_velocities = wp.zeros(MP, dtype=wp.vec3)
            self.angular_velocities = wp.zeros(MP, dtype=wp.vec3)
            self.inv_masses = wp.zeros(MP, dtype=float)
            self.inv_inertias = wp.zeros(MP, dtype=float)
            self.particle_masks = wp.zeros(MP, dtype=float)
            self.lengths = wp.zeros(MP, dtype=float)
            self.particle_radii = wp.zeros(MP, dtype=float)
            self.rod_radii = wp.zeros(MP, dtype=float)
            self.rest_basis = wp.zeros((MP, 3), dtype=wp.vec3)
            self.left_vertex_ids = wp.zeros(MP, dtype=int)

            hash_grid_dim = self.config.hash_grid_dim
            self.hash_grid = wp.HashGrid(hash_grid_dim, hash_grid_dim, hash_grid_dim)
            self.hash_grid_radius = self.config.collision_margin

            # ================== Mesh for Rendering ==================
            self.render_mesh_V = wp.zeros(200, dtype=wp.vec3)
            self.render_mesh_F = wp.zeros((300, 3), dtype=int)

    def _init_counters(self):
        self.n_particles = 0
        self.n_rods = 0
        self.n_bodies = 0
        self.n_body_shapes = 0

        # ================== Mesh for Rendering ==================
        self.n_render_vertices = 0
        self.n_render_faces = 0

    def _register_component_scene_get_id(self, comp: RodComponent):
        scene_ref = weakref.proxy(comp.entity.scene)
        if comp.entity.scene not in self.scenes:
            self.scenes.append(scene_ref)
        scene_id = self.scenes.index(scene_ref)

        component_id = len(self.components)
        self.components.append(weakref.proxy(comp))

        return scene_id, component_id

    def register_body_component(self, comp: BodyComponent):
        scene_id, component_id = self._register_component_scene_get_id(comp)
        comp.id_in_sys = component_id

        b_beg, b_end = self.n_bodies, self.n_bodies + 1
        self.n_bodies = b_end
        comp.body_id_in_sys = b_beg
        bs_beg, bs_end = self.n_body_shapes, self.n_body_shapes + comp.n_shapes
        self.n_body_shapes = bs_end

        # ================== Body Data ==================
        cm2world = comp.entity.pose * comp.cm2body
        wp_slice(self.body_q, b_beg, b_end).fill_(
            wp.transform(cm2world.p, np.concatenate((cm2world.q[1:], cm2world.q[:1])))
        )
        wp_slice(self.body_qd, b_beg, b_end).fill_(
            wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        wp_slice(self.body_f_ext, b_beg, b_end).fill_(0.0)

        # ================== Body Shape Data ==================
        wp_slice(self.body_shape_types, bs_beg, bs_end).assign(comp.shape_types)
        wp_slice(self.body_shape_scales, bs_beg, bs_end).assign(comp.scales)
        wp_slice(self.body_shape_fric, bs_beg, bs_end).assign(comp.frictions)
        wp_slice(self.body_shape_volumes, bs_beg, bs_end).assign(comp.volume_ids)
        wp_slice(self.body_shape_shape2cm, bs_beg, bs_end).assign(
            [
                wp.transform(
                    shape2cm.p, np.concatenate((shape2cm.q[1:], shape2cm.q[:1]))
                )
                for shape2cm in comp.shape2cm
            ]
        )
        wp_slice(self.body_shape_body_ids, bs_beg, bs_end).fill_(b_beg)

    def register_rod_component(self, comp: RodComponent):
        scene_id, component_id = self._register_component_scene_get_id(comp)
        comp.id_in_sys = component_id

        self.hash_grid_radius = max(
            self.hash_grid_radius,
            2.0 * (comp.length_per_rod + comp.radius)
            + self.config.collision_margin
            + 2.0 * self.config.max_velocity * self.config.time_step,
        )

        # ================== Particle Data ==================
        p_beg, p_end = self.n_particles, self.n_particles + comp.n_particles
        self.n_particles = p_end
        comp.particles_ptr_in_sys = (p_beg, p_end)
        wp_slice(self.positions, p_beg, p_end).assign(comp.positions)
        wp_slice(self.positions_prev_step, p_beg, p_end).assign(comp.positions)
        wp_slice(self.positions_rest, p_beg, p_end).assign(comp.positions)
        wp_slice(self.linear_velocities, p_beg, p_end).zero_()
        wp_slice(self.inv_masses, p_beg, p_end).assign(comp.inv_mass)
        wp_slice(self.particle_masks, p_beg, p_end).fill_(1.0)
        wp_slice(self.particle_radii, p_beg, p_end).fill_(comp.radius)

        # ================== Rod Data ==================
        r_beg, r_end = self.n_rods, self.n_rods + comp.n_rods
        self.n_rods = r_end
        comp.rods_ptr_in_sys = (r_beg, r_end)
        wp_slice(self.quaternions, r_beg, r_end).assign(comp.quaternions)
        wp_slice(self.quaternions_prev_step, r_beg, r_end).assign(comp.quaternions)
        wp_slice(self.angular_velocities, r_beg, r_end).zero_()
        wp_slice(self.inv_inertias, r_beg, r_end).assign(comp.inv_inertia)
        wp_slice(self.lengths, r_beg, r_end).assign(comp.lengths)
        wp_slice(self.rest_basis, r_beg, r_end).assign(comp.rest_basis)
        wp_slice(self.left_vertex_ids, r_beg, r_end).assign(np.arange(p_beg, p_end - 1))
        wp_slice(self.rod_radii, r_beg, r_end).fill_(comp.radius)

        # ================== Mesh for Rendering ==================
        render_v_beg = self.n_render_vertices
        render_v_end = self.n_render_vertices + comp.rod_mesh_V.shape[0]
        self.n_render_vertices = render_v_end
        comp.render_v_ptr_in_sys = (render_v_beg, render_v_end)
        render_f_beg = self.n_render_faces
        render_f_end = self.n_render_faces + comp.rod_mesh_F.shape[0]
        self.n_render_faces = render_f_end
        comp.render_f_ptr_in_sys = (render_f_beg, render_f_end)
        wp_slice(self.render_mesh_V, render_v_beg, render_v_end).assign(comp.rod_mesh_V)
        wp_slice(self.render_mesh_F, render_f_beg, render_f_end).assign(comp.rod_mesh_F)

    def update_render(self):
        for comp in self.components:
            if isinstance(comp, RodComponent):
                comp.update_render()

    def _init_step(self):
        wp.copy(self.positions_prev_step, self.positions, count=self.n_particles)
        wp.copy(self.quaternions_prev_step, self.quaternions, count=self.n_rods)

    def _kinematic_update(self):
        wp.launch(
            kernel=kinematic_update,
            dim=max(self.n_particles, self.n_rods),
            inputs=[
                self.n_particles,
                self.n_rods,
                self.positions,
                self.quaternions,
                self.linear_velocities,
                self.angular_velocities,
                self.particle_masks,
                self.config.gravity,
                self.config.time_step,
                self.config.damping,
            ],
        )

    def _compute_velocities(self):
        wp.launch(
            kernel=compute_velocities,
            dim=max(self.n_particles, self.n_rods),
            inputs=[
                self.n_particles,
                self.n_rods,
                self.positions,
                self.positions_prev_step,
                self.quaternions,
                self.quaternions_prev_step,
                self.linear_velocities,
                self.angular_velocities,
                self.config.time_step,
            ],
        )

    def _project_elastic_constraints(self):
        wp.launch(
            kernel=project_ss_constraints,
            dim=self.n_rods,
            inputs=[
                self.positions,
                self.quaternions,
                self.left_vertex_ids,
                self.inv_masses,
                self.inv_inertias,
                self.rest_basis,
                self.lengths,
                self.particle_masks,
                self.config.stretch_shear_stiffness,
                self.config.time_step,
            ],
        )
        assert self.n_rods >= 1
        wp.launch(
            kernel=project_bt_constraints,
            dim=self.n_rods - 1,
            inputs=[
                self.quaternions,
                self.left_vertex_ids,
                self.inv_inertias,
                self.lengths,
                self.config.bend_twist_stiffness,
                self.config.time_step,
            ],
        )

    def _build_hash_grid(self):
        wp.launch(
            kernel=compute_rod_centers,
            dim=self.n_rods,
            inputs=[
                self.positions,
                self.left_vertex_ids,
                self.rod_centers,
            ],
        )
        self.hash_grid.build(
            wp_slice(self.rod_centers, 0, self.n_rods),
            self.hash_grid_radius,
        )

    def _project_collision_constraints(self):
        wp.launch(
            kernel=project_collision_point_body,
            dim=self.n_particles,
            inputs=[
                self.n_body_shapes,
                self.particle_masks,
                self.positions,
                self.positions_prev_step,
                self.particle_radii,
                self.body_q,
                self.body_q_prev,
                self.body_shape_body_ids,
                self.body_shape_types,
                self.body_shape_scales,
                self.body_shape_volumes,
                self.body_shape_shape2cm,
                self.body_shape_fric,
                self.config.collision_margin,
                self.config.time_step,
            ],
        )
        wp.launch(
            kernel=project_collision_rod_rod,
            dim=self.n_rods,
            inputs=[
                self.hash_grid.id,
                self.positions,
                self.left_vertex_ids,
                self.inv_masses,
                self.rod_radii,
                self.config.collision_margin,
                self.config.time_step,
                self.hash_grid_radius,
            ]
        )

    def step(self):
        self._init_step()

        self._kinematic_update()

        self._build_hash_grid()

        for pbd_iter in range(self.config.n_pbd_iters):
            self._project_elastic_constraints()
            self._project_collision_constraints()

        self._compute_velocities()
