from setuptools import setup, find_packages

setup(
    name='pcd_metrics',
    version='0.1',
    packages=find_packages(),
    
    install_requires = [
        'numpy',
        'matplotlib',
        'tqdm',
        'mitsuba',
        'torch',
        'chamfer_distance',
        'lpips',
        'pyvista',
        'pyvirtualdisplay',
        'lpips',
        'ipython',
        'pytorch3d',
    ],

    dependency_links=[
        # Make sure to include the `#egg` portion so the `install_requires` recognizes the package
        "git+https://github.com/facebookresearch/pytorch3d.git@stable",
        "git+'https://github.com/otaheri/chamfer_distance'"
    ]
)