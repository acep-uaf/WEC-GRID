
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
ess.Vdc_nom = gsc.Vmag*sqrt(2)*1.25; %V, nominal dc bus voltage. set to 25% higher than grid voltage
ess.Vdc_0 = ess.Vdc_nom;
ess.Vdc_del = ess.Vdc_nom-gsc.Vmag*sqrt(2); %max deviation from nominal voltage. When determining this value, ensure (Vdc_nom - Vdc_del) > gsc.Vmag*sqrt(2) ?

%specify storage type and energy capacity
ess.storageType = "bat"; %"sc" for supercapacitor, "bat" for battery
% ess.storageType = "sc"; %"sc" for supercapacitor, "bat" for battery

ess.Ecap = 1e3; %total energy storage capacity, Wh

%determines the model parameters based on storage type, capacity, and
%nominal voltage
if ess.storageType == "bat"
    ess = init_ess_passive_bat(ess);

else %for now assume if not bat, do sc
    ess = init_ess_passive_sc(ess);

end

%PE and DC bus params, only applicable if isActive=1
ess.isActive = 0;
ess.Cdc = 100e-6;
ess.kp_v = 0;
ess.ki_v = 0;


%update dc bus max voltage deviation if storage type is more restrictive
ess.Vdc_del = min([ess.Vdc_del, (ess.Vdc_nom - ess.Vmin), (ess.Vmax - ess.Vdc_nom)]);





%% 
%plot(m2g_out.Pgen)
%figure()
%plot(ADCP_vel(:,2))
%plot( 
