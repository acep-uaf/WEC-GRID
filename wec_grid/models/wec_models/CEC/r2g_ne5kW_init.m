
%% Simulation Data        
simu.endTime = simLength;                     % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.1; 	

%% Synchronous Generator

mot.Pr=5e3; % Rated Power %% was 6e3
mot.Nr= 150; % Rated Speed in RPM
mot.Wr=2*pi*mot.Nr/60;
mot.Tr=mot.Pr/mot.Wr;
mot.Rs=2;%3.711; % Stator Resistance
mot.Lsd=45.25e-3; % Stator d-axis Inductance
mot.Lsq=64.88e-3; % Stator q-axis Inductance
mot.Ls = mot.Lsd;
mot.P=2;%24; % Number of poles
mot.Eb=440; % back emf
mot.k= mot.Eb/(150*2*pi/60)/sqrt(3); % machine constant (Vs/rad)
mot.lambda_fd=3*2*mot.k/mot.P; % Flux Constant
mot.lambda_sq_0 = 0;
mot.lambda_sd_0 = mot.lambda_fd;  % (recognizing that the d-axis is always aligned with the rotor magnetic axis                        
mot.J= 10.2564; % Inertia Constant kg/m^2
%Assume 1% friction loss 
mot.B=0;%0.01*mot.Pr/(mot.Wr^2); % Friction Co-efficient
mot.theta_d_0 = 0;


%% Back-to-back converter parameters

%WEC-side converter
wsc.Bdamp = 750/(90*pi/30)^2;%5e3/(90*pi/30)^2 %damping coeff. Bdamp*angular_speed = Torque, so Bdamp*(max_angular_speed)^2 = 5kW
wsc.Fpto_lim = 5e3/(90*pi/30)*0.99; %this is torque in the sim. power/max_angular_speed = max_torque
%NOTE: the Fpto limit is reduced to 99.9% to avoid a singularity in the
%simulation. Theoretically this shouldn't be needed but I think it happens
%due to small error accumulation from numerical calculations

%grid-side converter
gsc.Prated = 10e3;
gsc.Vmag = 480*1.1; %V, rms, l-l, 10% higher voltage than grid Vnom
gsc.Ilim = gsc.Prated/gsc.Vmag; %A, rms
gsc.Tavg = 5*60; %averaging period, s

%voltage correction PI controller
gsc.kp = gsc.Prated;
gsc.ki = 0;


%% Onboard energy storage
%energy storage system
ess.Vdc_0 = gsc.Vmag*sqrt(2)*1.25; %V, nominal dc bus voltage. set to 25% higher than grid voltage
ess.Vdc_del = ess.Vdc_0-gsc.Vmag*sqrt(2); %max deviation from nominal voltage. When determining this value, ensure (Vdc_nom - Vdc_del) > gsc.Vmag*sqrt(2) ?

%specify storage type and energy capacity
ess.storageType = "bat"; %"sc" for supercapacitor, "bat" for battery
ess.Ecap = 1e3; %total energy storage capacity, Wh

%determines the model parameters based on storage type, capacity, and
%nominal voltage
%battery cell parameters and limits
Vcell_max = 4.2; %V
Vcell_min = 3.0; %V
SOC_max = 1.0; %SOC associated with Vcell_max
SOC_min = 0.2; %SOC associated with Vcell_min
% Vcell_nom = (Vcell_max + Vcell_min)/2; %nominal cell voltage, V
Vcell_nom = (Vcell_max - Vcell_min)/(SOC_max - SOC_min)/2 + Vcell_min; %nominal cell voltage, V

%determine max and min battery pack voltage levels
%assume that ess.Vdc_0 is the nominal voltage of the battery 
% pack (i.e., when SOC = 70% in this model)
ess.Vmax = ess.Vdc_0*Vcell_max/Vcell_nom;
ess.Vmin = ess.Vdc_0*Vcell_min/Vcell_nom;

ess.SOC_LUT = 0:0.01:1; %range of SOC values in lookup table

%assume a linear voltage vs SOC curve, 
% where Vmin occurs when SOC = SOC_min and Vmax occurs when SOC = SOC_max
ess.Voc_LUT = (ess.SOC_LUT - SOC_min)*(ess.Vmax - ess.Vmin)/(SOC_max - SOC_min) + ess.Vmin; 

Ah_cap = ess.Ecap/ess.Vdc_0; %Amp-hour capacity of battery
ess.Csoc = Ah_cap*3600; %SOC capacitance, A-s
% in an electrical equivalent circuit, voltage across this "capacitor" is 
% equivalent to the SOC.

ess.SOC_0 = 0.70; %initial SOC of bat, set to Vdc_nom

ess.eff = 0.90; %efficiency of bat

%update dc bus max voltage deviation if storage type is more restrictive
ess.Vdc_del = min([ess.Vdc_del, (ess.Vdc_0 - ess.Vmin), (ess.Vmax - ess.Vdc_0)]);

%% 
%plot(m2g_out.Pgen)
%figure()
%plot(ADCP_vel(:,2))
%plot( 
