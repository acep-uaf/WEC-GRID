clear, close all, format compact

%w2g inputs
wecId = 1;
simLength = 1*3600; %[s]
Tsample = 5*60; %[s]
waveHeight = 2.5; %[m]
wavePeriod = 8; %[s]
% waveSeed = 2.2; %integer


%run
% [m2g_out] = WEC2G_sim(wecId,simLength,Tsample,waveHeight,wavePeriod,waveSeed);
[m2g_out] = WEC2G_sim(wecId,simLength,Tsample,waveHeight,wavePeriod);