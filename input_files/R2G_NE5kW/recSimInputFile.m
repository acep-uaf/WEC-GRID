%% Simulation Data

%load river speed data
%Currently just using power data and scaling it down some. Need to get a
%river current profile sample
load("NE5kW_sample_data.mat")
vel_sw.time = t;
vel_sw.signals.values = PElec/1e3;


%% Sim params
simu = simulationClass();  
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 0;                      % Wave Ramp Time [s]
simu.endTime = vel_sw.time(end);             % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.1; 	


%% Turbine params

load('LUT_powerCurve_NE5kW.mat')
LUT.u = LUT_vel;
LUT.Pgen = LUT_Pgen;


%% Back-to-back converter parameters

%grid-side converter
gsc.Prated = 5e3;
gsc.Vmag = 480*1.1; %V, rms, l-l, 10% higher voltage than grid Vnom
gsc.Ilim = gsc.Prated/gsc.Vmag; %A, rms
gsc.Tavg = 5*60; %averaging period, s

%voltage correction PI controller
gsc.kp = gsc.Prated;
gsc.ki = 0;


%% Onboard energy storage

%energy storage system
ess.Vdc_0 = gsc.Vmag*sqrt(2)*1.25; %V, 25% higher than grid voltage
ess.C = 8; %F
ess.Vdc_del = ess.Vdc_0-gsc.Vmag*sqrt(2); %max deviation from nominal voltage. When determining this value, ensure (Vdc_nom - Vdc_del) > gsc.Vmag*sqrt(2) ?





