
# ![WEC-GRID Logo](docs/wec-grid-logo.png)

## WEC-GRID: Integrating Wave Energy Converters into Power Grid Simulations

**WEC-GRID** is an open-source Python library crafted to simulate the integration of Wave Energy Converters (WECs) and Current Energy Converters (CECs) into renowned power grid simulators like [PSS®E](https://new.siemens.com/global/en/products/energy/services/transmission-distribution-smart-grid/consulting-and-planning/pss-software/pss-e.html) & [PyPSA](https://pypsa.org/).

### Introduction

Amidst the global shift towards sustainable energy solutions, Wave Energy Converters (WECs) and Current Energy Converters (CECs) emerge as groundbreaking innovations. These tools harbor the potential to tap into the boundless energy reserves of our oceans. Yet, to weave them into intricate systems like microgrids, a profound modeling, testing, and analysis regimen is indispensable. WEC-GRID, presented through this Jupyter notebook, is a beacon of both demonstration and guidance, capitalizing on an open-source software to transcend these integration impediments.

### Overview

![WEC-GRID Data Flowchart](docs/WecGrid-flowchart.png)

WEC-GRID is in its nascent stages, yet it presents a Python Jupyter Notebook that successfully establishes a PSSe API connection. It can solve both static AC & DC power flows, injecting data from a WEC/CEC device. Additionally, WEC-GRID comes equipped with rudimentary formatting tools for data analytics. The modular design ensures support for a selected power flow solving software and WEC/CEC devices. 

For the current implementations, WEC-GRID is compatible with PSSe and [WEC-SIM](https://wec-sim.github.io/WEC-Sim/). The widespread application of PSSe in the power systems industry, coupled with its robust API, makes it an ideal choice.

### Software Requirements and Setup

#### Installation Sequence:
1. PSSe
2. Miniconda
3. Conda packages
4. Matlab 2021
5. Matlab engine

#### Instructions:
1. Use the [conda](https://docs.conda.io/en/latest/miniconda.html) package manager (64 bit) to install supporting software.
2. During the Miniconda setup, ensure you revert to pywin version 228:
```bash
pip install pywin32==228
```
Note: Miniconda on Windows might pose some challenges, and results can vary.

##### Environment Setup
```bash
conda create --name PSSe python=3.7 
conda activate PSSe
conda install pandas
conda install matplotlib
conda install Jupyter
```
#### MATLAB Integration
For MATLAB integration into your environment, follow the instructions [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Ensure you're in your PSSe conda environment during installation.

