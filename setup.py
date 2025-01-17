from setuptools import setup, find_packages

setup(
    name="WEC_GRID",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        "numpy>=1.18.5",
        "pandas>=1.0.4",
        "matplotlib>=3.1.0",
        "seaborn>=0.11.0",
        "pypsa",
        "pypower",
        "pyrlu",
        "ipycytoscape",
        "bqplot",
        "pywin32>=228; platform_system=='Windows'",  # Conditional dependency for Windows
        "ipykernel>=6.0",
    ],
    description=(
        "WEC-GRID is an open-source Python library crafted to simulate the integration "
        "of Wave Energy Converters (WECs) and Current Energy Converters (CECs) into "
        "renowned power grid simulators like PSS®E & PyPSA."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander Barajas-Ritchie",
    author_email="barajale@oregonstate.edu",
    url="https://github.com/acep-uaf/WEC-GRID",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Adjust based on the minimum Python version your library supports
)