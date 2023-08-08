clear, close all, format compact

%w2g inputs
wecId = 1;
simLength = 1*3600; %[s]
Tsample = 5*60; %[s]
waveHeight = 2.5; %[m]
wavePeriod = 8; %[s]
% waveSeed = 2.2; %integer


%run
% [m2g_out] = w2gSim_LUPA(wecId,simLength,Tsample,waveHeight,wavePeriod,waveSeed);
[m2g_out] = w2gSim_LUPA(wecId,simLength,Tsample,waveHeight,wavePeriod);