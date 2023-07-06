from setuptools import setup, find_packages

# Function to parse requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = (line.strip().split('=')[:2] for line in f)
        return [''.join(line) for line in lines if line and not line[0].startswith('#')]

requirements = parse_requirements('requirements.txt')

setup(
    name='wec_grid',
    version='0.1.0',
    packages=find_packages(),  # automatically find all packages
    install_requires=requirements,  # dependencies from requirements.txt
    entry_points={
        'console_scripts': [
            'wec_grid = wec_grid.wec_grid:main',  # If you have a main function to run your script
        ]
    },
)
