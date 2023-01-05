# WEC-GRID
WEC-GRID is a Python library for simulating Wave Energy Converter on PSSe


## Installation

Install order: PSSe, miniconda, conda packages, Matlab 2021, Matlab engine

Use the mini conda package manager (64 bit)[conda](https://docs.conda.io/en/latest/miniconda.html) to install supporting softwares

at the miniconda step, revert pywin to version 228.
'''bash
pip install pywin32==228
'''
miniconda on windows seems to be tricky. mileage will vary

### Enviroment set up

Use miniconda 64 bit to create enviroment: use the command below from the WEC-GRID directory

```bash
conda create --name PSSe --file .\spec-file.txt
```
if that doesn't work run these commands:

```bash
conda create --name PSSe python=3.7 
```
```bash
conda activate PSSe
```
```bash
conda install pandas
```
```bash
conda install matplotlib
```
```bash
conda install Jupyter
```

#### Matlab
You'll need to add Matlab to your enviroment as well, use this resource [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)


```bash
cd C:\Program Files\MATLAB\R2021b\extern\engines\python
```
```bash
 python setup.py install
```
*make sure you're in you're PSSe conda enviroment
