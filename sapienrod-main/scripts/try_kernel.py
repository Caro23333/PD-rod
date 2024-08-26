import warp as wp
import numpy as np

from sapienrod.kernels.distance_funcs import edge_egde_closest_point

wp.init()


@wp.func
def func_two_returns():
    a = int(1)
    b = int(2)
    b += 5
    return a, b


@wp.kernel
def test_kernel():
    a, b = func_two_returns()
    wp.printf("a: %d, b: %d\n", a, b)


@wp.kernel
def test_distance_kernel(edges: wp.array(dtype=wp.vec3, ndim=2)):
    i = wp.tid()
    t0, t1 = edge_egde_closest_point(edges[i, 0], edges[i, 1], edges[i, 2], edges[i, 3])
    wp.printf("t0: %f, t1: %f\n", t0, t1)


device = "cuda:0"
# wp.launch(kernel=test_kernel, dim=1, inputs=[], outputs=[], device=device)

edges = wp.array(
    np.array(
        [
            [
                [0.006900, 0.006900, 2.049990],
                [0.007000, 0.007000, 1.999990],
                [0.006800, 0.006800, 2.099990],
                [0.006900, 0.006900, 2.049990],
            ]
        ]
    ),
    dtype=wp.vec3,
    device=device,
)
wp.launch(kernel=test_distance_kernel, dim=1, inputs=[edges], device=device)
wp.synchronize()
