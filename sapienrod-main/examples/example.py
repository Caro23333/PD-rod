import warp as wp
import numpy as np
import sapien
import trimesh
from PIL import Image
import os

from sapienrod.rod_component import *
from sapienrod.rod_system import RodSystem, RodConfig
from sapienrod.utils.array import wp_slice
from sapienrod.rod_defs import ShapeTypes


def init_camera(scene: sapien.Scene):
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(1024, 1024)
    cam.set_near(1e-3)
    cam.set_far(1000)
    cam_entity.add_component(cam)
    cam_entity.name = "camera"
    cam_entity.set_pose(
        # sapien.Pose([-1, 0.5, 1], [1, 0, 0, 0])
        sapien.Pose([-0.472884, 0.014437, 0.33941], [0.98148, 0.00191307, 0.191304, -0.00981516])
    )
    scene.add_entity(cam_entity)
    return cam


def main():
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
    scene.add_ground(0.0)

    cam = init_camera(scene)

    config = RodConfig()
    system = RodSystem(config=config, device="cuda:0")
    scene.add_system(system)
    
    # plane_comp = BodyComponent(
    #     shape_types=[ShapeTypes.GEO_PLANE],
    # )
    # plane_render = sapien.render.RenderBodyComponent()
    # plane_render.attach(
    #     sapien.render.RenderShapePlane(
    #         np.array([10., 10., 10.]),
    #         sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
    #     )
    # )
    # plane_entity = sapien.Entity()
    # plane_entity.add_component(plane_comp)
    # plane_entity.add_component(plane_render)
    # plane_entity.set_pose(sapien.Pose(q=[0.7071068, 0, -0.7071068, 0]))
    # scene.add_entity(plane_entity)

    box_radius = 0.2
    box_comp = BodyComponent(
        shape_types=[ShapeTypes.GEO_PLANE] * 5,
        cm2body=sapien.Pose(),
        shape2cm=[
            sapien.Pose(q=[0.7071068, 0, -0.7071068, 0]),  # ground
            sapien.Pose(p=[-box_radius, 0, 0]),  # face x axis
            sapien.Pose(p=[0, -box_radius, 0], q=[ 0.7071068, 0, 0, 0.7071068 ]),  # face y axis
            sapien.Pose(p=[box_radius, 0, 0], q=[ 0, 0, 0, 1 ]),  # face -x axis
            sapien.Pose(p=[0, box_radius, 0], q=[ -0.7071068, 0, 0, 0.7071068 ]),  # face -y axis
        ]
    )
    box_entity = sapien.Entity()
    box_entity.add_component(box_comp)
    scene.add_entity(box_entity)

    rod_mesh = trimesh.load_mesh(
        "/home/xiaodi/Desktop/Repos/sapienrod/assets/cylinder.obj"
    )

    height = 0.5
    rod_comp = RodComponent(
        n_rods=100,
        # start=np.array([0, 0, height], dtype=float),
        # end=np.array([0, 1, height], dtype=float),
        start=np.array([0, 0, 5 + height], dtype=float),
        end=np.array([0.01, 0.01, height], dtype=float),
        radius=0.01,
        rod_mesh_V=rod_mesh.vertices,
        rod_mesh_F=rod_mesh.faces,
    )
    rod_render = rod_comp.create_render_component()

    rod_entity = sapien.Entity()
    rod_entity.add_component(rod_comp)
    rod_entity.add_component(rod_render)
    scene.add_entity(rod_entity)
    
    # wp_slice(system.particle_masks, 0, 1).zero_()

    viewer = sapien.utils.Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_pose(cam.get_entity_pose())
    viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)
    viewer.paused = True

    system.update_render()
    viewer.render()
    
    n_time_steps = 10000
    render_every = 10
    save_render = False
    save_render = True
    
    if save_render:
        save_dir = "/home/xiaodi/Desktop/Repos/sapienrod/output/example_kbb2e-5"
        os.makedirs(save_dir, exist_ok=True)
    
    for step in range(n_time_steps):
        system.step()
        if step % render_every == 0:
            system.update_render()
            scene.update_render()
            viewer.render()
            
            if save_render:
                cam.take_picture()
                rgba = cam.get_picture("Color")
                rgba = np.clip(rgba, 0, 1)[:, :, :3]
                rgba = Image.fromarray((rgba * 255).astype(np.uint8))
                rgba.save(os.path.join(save_dir, f"step_{(step + 1) // render_every:04d}.png"))


if __name__ == "__main__":
    main()
