clear, close all, format compact

%Set the sampling period of the time series to be sent to the power flow
%solver
Ts = 5*60; %s

%% Initialization

%initialize
recSimInputFile

%% Run W2G sim

sim("R2G_ss_NE5kW.slx");


%% Post-Processing

%downsample data to desired resolution for PSSE
Pgrid_ds = DownSampleTS(m2g_out.Pgrid,Ts,1);
Qgrid_lim_ds = DownSampleTS(m2g_out.Qgrid_lim,Ts,1);


%% Plots

plot_R2G_waveforms


