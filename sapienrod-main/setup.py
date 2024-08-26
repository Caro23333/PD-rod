from setuptools import setup, find_packages

setup(
    name='sapienrod',
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    install_requires=[
        'numpy',
        'meshio',
        'scipy',
    ],
)
