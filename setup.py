from setuptools import setup, find_packages

setup(
    name='wec_grid',
    version='0.1.0',
    packages=find_packages(),  # automatically find all packages
    install_requires=[],  # List all dependencies here
    entry_points={
        'console_scripts': [
            'wec_grid = wec_grid.wec_grid:main',  # If you have a main function to run your script
        ]
    },
)
