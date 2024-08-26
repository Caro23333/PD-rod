import warp as wp
import numpy as np
import sapien
import typing
from typing import List, Union
from sapien.render import RenderCudaMeshComponent
from .utils.array import wp_slice
from .kernels.render_kernels import *

if typing.TYPE_CHECKING:
    from .rod_system import RodSystem


class BodyComponent(sapien.Component):
    def __init__(
        self,
        shape_types: List[int],
        scales: Union[np.ndarray, List[float]] = None,
        frictions: Union[np.ndarray, List[float]] = None,
        volumes: List[wp.Volume] = None,
        cm2body: sapien.Pose = None,
        shape2cm: List[sapien.Pose] = None,
        source: sapien.physx.PhysxRigidBodyComponent = None,
    ):
        super().__init__()
        self.source = source

        self.id_in_sys = None  # component id
        self.body_id_in_sys = None
        self.q_slice = None
        self.qd_slice = None  # (omega, v) in the world frame
        self.f_ext_slice = None  # (tau, f) in the world frame

        self.n_shapes = len(shape_types)
        self.shape_types = shape_types
        self.scales = scales if scales is not None else np.ones((self.n_shapes, 3))
        self.frictions = frictions if frictions is not None else np.zeros(self.n_shapes)
        self.volumes = volumes
        self.volume_ids = (
            [v.id if v else 0 for v in volumes]
            if volumes is not None
            else [0] * self.n_shapes
        )
        self.cm2body = cm2body if cm2body is not None else sapien.Pose()
        self.shape2cm = (
            shape2cm if shape2cm is not None else [sapien.Pose()] * self.n_shapes
        )

        assert (
            len(self.scales) == self.n_shapes
        ), f"scales length mismatch: got {self.n_shapes} shape_types but {len(self.scales)} scales"
        assert (
            len(self.frictions) == self.n_shapes
        ), f"frictions length mismatch: got {self.n_shapes} shape_types but {len(self.frictions)} frictions"
        assert (
            len(self.volume_ids) == self.n_shapes
        ), f"volume_ids length mismatch: got {self.n_shapes} shape_types but {len(self.volume_ids)} volume_ids"
        assert (
            len(self.shape2cm) == self.n_shapes
        ), f"shape2cm length mismatch: got {self.n_shapes} shape_types but {len(self.shape2cm)} shape2cm"

    def on_add_to_scene(self, scene: sapien.Scene):
        s: RodSystem = scene.get_system("rod")
        s.register_body_component(self)


class RodComponent(sapien.Component):
    def __init__(
        self,
        n_rods: int = 10,
        radius: float = 0.01,
        start: np.ndarray = np.array([0, 0, 0]),
        end: np.ndarray = np.array([0, 1, 0]),
        density: float = 1e3,
        kss: float = 1e3,  # stretching and shearing stiffness
        kbt: float = 1e0,  # bending and twisting stiffness
        friction: float = 0.0,
        rod_mesh_V: np.ndarray = None,  # mesh for each rod, used for rendering
        rod_mesh_F: np.ndarray = None,
    ):
        super().__init__()

        self.n_rods = n_rods
        self.n_particles = n_rods + 1

        self.id_in_sys = None
        self.particles_ptr_in_sys = None
        self.rods_ptr_in_sys = None
        self.render_v_ptr_in_sys = None
        self.render_f_ptr_in_sys = None

        self.radius = radius  # only used for mass calculation
        self.density = density
        self.kss = kss  # stretching and shearing
        self.kbt = kbt  # bending and twisting
        self.friction = friction

        e3 = end - start  # rest axis along the rod
        length = np.linalg.norm(e3)
        assert length > 1e-9, f"Rod length {length} is too small"
        e3 /= length
        e2 = np.cross(e3, np.array([1, 0, 0]))
        if np.linalg.norm(e2) < 1e-9:
            e2 = np.cross(e3, np.array([0, 1, 0]))
        e2 /= np.linalg.norm(e2)
        e1 = np.cross(e2, e3)

        self.positions = np.linspace(start, end, self.n_particles)
        self.quaternions = np.zeros((self.n_rods, 4))  # [x, y, z, w]
        self.quaternions[:, -1] = 1
        self.rest_basis = np.zeros((self.n_rods, 3, 3), dtype=float)  # rest axes
        self.rest_basis[:, 0] = e1
        self.rest_basis[:, 1] = e2
        self.rest_basis[:, 2] = e3

        self.length_per_rod = length / n_rods
        mass_per_rod = np.pi * radius**2 * self.length_per_rod * density
        self.lengths = np.ones(self.n_rods, dtype=float) * self.length_per_rod
        self.inv_mass = np.ones(self.n_particles, dtype=float) / mass_per_rod
        self.inv_mass[0] = 2.0 / mass_per_rod
        self.inv_mass[-1] = 2.0 / mass_per_rod
        # TODO: how to properly set the inertia?
        self.inv_inertia = np.ones(self.n_rods, dtype=float) / (
            mass_per_rod * self.length_per_rod**2
        )

        # ================== Mesh for Rendering (Temporary) ==================
        # assert rod_mesh_V is not None and rod_mesh_F is not None
        # n_mesh_V = rod_mesh_V.shape[0]
        # rod_mesh_V *= np.array([radius, radius, length_per_rod])
        # rod_mesh_V = np.matmul(rod_mesh_V[None, :, :], self.rest_basis)  # [n_rods, N, 3]
        # rod_centers = (self.positions[:-1] + self.positions[1:]) / 2  # [n_rods, 3]
        # rod_mesh_V += rod_centers[:, None, :]
        # rod_mesh_F = np.tile(rod_mesh_F[None, :, :], (self.n_rods, 1, 1))  # [n_rods, N, 3]
        # rod_mesh_F += np.arange(0, self.n_rods)[:, None, None] * n_mesh_V
        # self.rod_mesh_V = rod_mesh_V.reshape(-1, 3)
        # self.rod_mesh_F = rod_mesh_F.reshape(-1, 3)

        # rod_mesh_V *= np.array([radius, radius, length_per_rod])
        self.rod_mesh_V = rod_mesh_V
        self.rod_mesh_F = rod_mesh_F

    def on_add_to_scene(self, scene: sapien.Scene):
        s: RodSystem = scene.get_system("rod")
        s.register_rod_component(self)

    def create_render_component(self):
        n_render_V = self.rod_mesh_V.shape[0] * (self.n_rods + self.n_particles)
        n_render_F = self.rod_mesh_F.shape[0] * (self.n_rods + self.n_particles)
        render_F = (
            self.rod_mesh_F
            + np.arange(0, self.n_rods + self.n_particles)[:, None, None]
            * self.rod_mesh_V.shape[0]
        )
        render_comp = sapien.render.RenderCudaMeshComponent(n_render_V, n_render_F)
        render_comp.set_vertex_count(n_render_V)
        render_comp.set_triangle_count(n_render_F)
        render_comp.set_triangles(render_F.reshape(-1, 3))
        rod_color = [0.7, 0.3, 0.4, 1.0]
        render_comp.set_material(sapien.render.RenderMaterial(base_color=rod_color))
        return render_comp

    def update_render(self):
        ent = self.entity
        render_comp: RenderCudaMeshComponent = ent.find_component_by_type(
            RenderCudaMeshComponent
        )
        assert render_comp is not None
        s: RodSystem = self.entity.scene.get_system("rod")
        with wp.ScopedDevice(s.device):
            interface = render_comp.cuda_vertices.__cuda_array_interface__
            dst = wp.array(
                ptr=interface["data"][0],
                dtype=wp.float32,
                shape=interface["shape"],
                strides=interface["strides"],
                owner=False,
            )
            wp.launch(
                kernel=update_rod_meshes,
                dim=self.n_rods,
                inputs=[
                    self.rods_ptr_in_sys[0],
                    s.positions,
                    s.quaternions,
                    s.rest_basis,
                    s.left_vertex_ids,
                    s.render_mesh_V,
                    self.render_v_ptr_in_sys[0],
                    len(self.rod_mesh_V),
                    self.radius,
                ],
                outputs=[dst],
            )
            # wp.launch(
            #     kernel=update_particle_meshes,
            #     dim=self.n_particles,
            #     inputs=[
            #         self.particles_ptr_in_sys[0],
            #         s.positions,
            #         s.render_mesh_V,
            #         self.render_v_ptr_in_sys[0],
            #         len(self.rod_mesh_V),
            #         self.radius,
            #         self.n_rods * len(self.rod_mesh_V)
            #     ],
            #     outputs=[dst],
            # )
            render_comp.notify_vertex_updated(wp.get_stream().cuda_stream)
