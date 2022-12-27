# WEC-GRID

Work in progress.

Install order: PSSe, miniconda, conda packages, Matlab, Matlab engine
	- at the miniconda step, revert pywin to version 228. 
	- miniconda on windows seems to be tricky. mileage will vary

install PSSe here -> https://web.engr.oregonstate.edu/~barajale/
	- follow the suggested PSSe config

Enviroment:

Use miniconda 64 bit to create enviroment: use the command below from the WEC-GRID directory
    conda create --name PSSe --file .\spec-file.txt

if that doesn't work run these commands:
    conda create --name PSSe python=3.7 
    conda activate PSSe
    conda install pandasi
    conda install matplotlib
    conda install Jupyter

You'll need to add Matlab to your enviroment as well, use this resources 
    -> https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
*make sure you're using MATLAB 2021 or older

Matlab engine install

make your way to this directory -> C:\Program Files\MATLAB\R2021b\extern\engines\python
then run this command while being in your conda env -> python setup.py install 



