clear, close all, format compact

%Set the sampling period of the time series to be sent to the power flow
%solver
Ts = 5*60; %s


%% Initialization of WEC-Sim

%makes 'initializeWecSim' call 'wecSimInputFile' instead of other options
runWecSimCML = 1;

%initialize
run('initializeWecSim');


%% Run W2G sim
sim(simu.simMechanicsFile, [], simset('SrcWorkspace','parent'));


%% Post-Processing

enableUserDefinedFunctions = 0; %set whether the UDFs are called, 0 to not call

%run post-sim script
run('stopWecSim');

%downsample data to desired resolution for PSSE
Pgrid_ds = DownSampleTS(m2g_out.Pgrid,Ts,1);
Qgrid_lim_ds = DownSampleTS(m2g_out.Qgrid_lim,Ts,1);
 

%% Plots

plot_W2G_waveforms



