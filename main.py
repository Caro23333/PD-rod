from sim import Sim
from rod import Rod, getRotation, getRotation_low
from solver import *
import sapien as sapien
from sapien.utils import Viewer
import warp as wp
import time

wp.init()

if __name__ == "__main__":
    n = 120
    start = wp.vec3(-1.5, 0, 0.0)
    end = wp.vec3(1.5, 0, 0.0)
    radius = wp.length(end - start) / n
    rod = Rod(n, radius = radius, E = 3 * 10 ** 6, nu = 0.2, m = 600, start = start, end = end)
    sim = Sim(fps = 30, stepPerFrame = 25, rod = rod, solverIter = 10, wSE = 8 * 10 ** 3, wBT = 3 * 10 ** 2)
    sim.addTable(wp.vec3(-2.0, -2.0, -2.0), wp.vec3(-2.0, 2.0, -2.0),
                 wp.vec3(2.0, 2.0, -2.0), wp.vec3(2.0, -2.0, -2.0))
    sim.addTable(wp.vec3(-2.0, -2.0, -0.2), wp.vec3(-2.0, 0.25, -0.2),
                 wp.vec3(0.25, 0.25, -0.2), wp.vec3(0.25, -2.0, -0.2))
    sim.addTable(wp.vec3(-2.0, 0.25, -0.2), wp.vec3(-2.0, 2.0, -0.2),
                 wp.vec3(0.75, 2.0, -0.2), wp.vec3(0.75, 0.25, -0.2))
    sim.addTable(wp.vec3(0.75, -0.25, -0.2), wp.vec3(0.75, 2.0, -0.2),
                 wp.vec3(2.0, 2.0, -0.2), wp.vec3(2.0, -0.25, -0.2))
    sim.addTable(wp.vec3(0.25, -2.0, -0.2), wp.vec3(0.25, -0.25, -0.2),
                 wp.vec3(2.0, -0.25, -0.2), wp.vec3(2.0, -2.0, -0.2))
    sim.initBoundary()
    maxTimeSteps = 30000
    totalComputationTime = 0

    scene = sapien.Scene()

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer() 
    viewer.set_camera_xyz(x=0.5, y=-2.5, z=-0.35)

    viewer.set_camera_rpy(r=0, p=0, y=-3.1415926 / 2)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1.25)
    actorBuilder = scene.create_actor_builder()
    boxList = []
    visual_radius = wp.length(end - start) * 0.01
    for i in range(n):
        actorBuilder.add_box_visual(half_size = [radius, radius, 3.5 / (2 * n)], material = [i / n, 0.4, 0.4])
        boxList.append(actorBuilder.build(name = "box_" + str(i)))

    for i in range(maxTimeSteps):
        startTimer = time.time()
        sim.timeStep(fExt = wp.zeros(3 * n + 3, dtype = float), tExt = wp.zeros(3 * n + 3, dtype = float))
        if i % 10 == 0:
            pos = sim.rod.x.numpy()
            orient = sim.rod.u.numpy()
            for j in range(n):
                p = [(pos[3 * j] + pos[3 * j + 3]) / 2,
                    (pos[3 * j + 1] + pos[3 * j + 4]) / 2,
                    (pos[3 * j + 2] + pos[3 * j + 5]) / 2]
                q = getRotation_low(wp.vec3(pos[3 * j + 3] - pos[3 * j], 
                                    pos[3 * j + 4] - pos[3 * j + 1],
                                    pos[3 * j + 2] - pos[3 * j + 5]), initAxis = wp.vec3(0, 0, 1))
                boxList[j].set_pose(sapien.Pose(p = p, q = q))

            scene.update_render()  
            viewer.render()
        
        totalComputationTime += time.time() - startTimer
        print("********************** {} / {} steps, {:.2f} second cost".format(i + 1, maxTimeSteps, totalComputationTime))
    