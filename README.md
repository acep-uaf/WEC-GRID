# WEC-GRID

Work in progress.

Enviroment:

Use conda to create enviroment: use the command below from the WEC-GRID directory
    conda create --name PSSe --file .\spec-file.txt

if that doesn't work run these commands:
    conda create --name PSSe python=3.7
    conda activate PSSe
    conda install pandas
    conda install matplotlib
    conda install Jupyter

You'll need to add Matlab to your enviroment as well, use this resources 
    -> https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
*make sure you're using MATLAB 2021 or older

Matlab engine install

make your way to this directory -> C:\Program Files\MATLAB\R2021b\extern\engines\python
then run this command while being in your conda env -> python setup.py install 



