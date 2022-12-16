clear, close all, format compact


%% Initialization

%makes 'initializeWecSim' call 'wecSimInputFile' instead of other options
runWecSimCML = 1;

%initialize
run('initializeWecSim')


%% Run W2G sim
sim(simu.simMechanicsFile, [], simset('SrcWorkspace','current'));


%% Post-Processing

enableUserDefinedFunctions = 0; %set whether the UDFs are called, 0 to not call

%run post-sim script
%run('stopWecSim')

%plot_W2G_waveforms