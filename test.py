import warp as wp
import numpy as np
import warp.sparse as sparse
import os
import warp as wp
import time

wp.init()

@wp.kernel
def modfiy(s: float):
    s = 2.5

s = 0.0
wp.launch(modfiy, dim = 1, inputs = [s])
print(s)