# PD-rod

Cosserat rod with projective dynamics. This is a CUDA implementation based on _NVIDIA warp_; Visualization is done by [SAPIEN](https://github.com/haosulab/SAPIEN).

To use: In Linux, run `python3 main.py`.

What is included:
- Single elastic rod, reacting to arbitrary external forces and a variety of deformation (stretching / shearing / twisting / bending)
- Dirichlet boundary conditions on position and orientation
- Self-collision and collision with sphere / four-sided prism (but no visualization for latter two types of objects)
- Export checkpoints into `.obj` files.

What is not included so far:
- Multiple rods
- Frictional contact
- Contact between rod and stiff bodies with general SDF

### References
[Soler, C., Martin, T., & Sorkine‐Hornung, O. (2018, December). Cosserat rods with projective dynamics. In Computer Graphics Forum (Vol. 37, No. 8, pp. 137-147)](https://igl.ethz.ch/projects/cosserat-rods/CosseratRods-SCA2018.pdf)
