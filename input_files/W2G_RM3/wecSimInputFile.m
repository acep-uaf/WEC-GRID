%% Simulation Data
simu = simulationClass();        
simu.simMechanicsFile = 'W2G_ss_RM3.slx';      %Location of Simulink Model File with PTO-Sim  
simu.explorer = 'off';                  % Turn SimMechanics Explorer (on/off)
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 0;                      % Wave Ramp Time [s]
simu.endTime = 60*10;                     % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.1; 	


%% Wave Information
% Regular Waves  
% waves = waveClass('regular');            
% waves.height = 2.5;                          
% waves.period = 8;  

%Irregular Waves using PM Spectrum
waves = waveClass('irregular');
waves.height = 2.5;
waves.period = 8;
waves.spectrumType = 'PM';
waves.phaseSeed= 2;

%The equal energy formulation speeds up the irregular wave simulation time
% by reducing the number of frequencies the wave train is defined by,
% and thus the number of frequencies for which the wave forces are
% calculated. It prevents bins with very little energy from being created
% and unnecessarily adding to the computational cost.
% waves.bem.option = 'EqualEnergy';


%% Body Data
% Float
body(1) = bodyClass('./hydroData/rm3.h5');             
body(1).geometryFile = './geometry/float.stl';      
body(1).mass = 'equilibrium';                   
body(1).inertia = [20907301 21306090.66 37085481.11];     
body(1).quadDrag.cd = ([1 1 1 0 0 0]);
body(1).quadDrag.area = ([5*20 5*20 (14^2)*pi 0 0 0]);


% Spar/Plate
body(2) = bodyClass('./hydroData/rm3.h5');     
body(2).geometryFile = './geometry/plate.stl';  
body(2).mass = 'equilibrium';                   
body(2).inertia = [94419614.57 94407091.24 28542224.82];
body(2).quadDrag.cd = ([2.56 2.56 3.21 0 0 0]);
body(2).quadDrag.area = ([(0.1*30+6*38) (0.1*30+6*38) (15^2)*pi 0 0 0]);


%% PTO and Constraint Parameters
% Translational Constraint
constraint(1) = constraintClass('Constraint1'); 
constraint(1).location = [0 0 0]; 

% Translational PTO
pto(1) = ptoClass('PTO1');           	% Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                           % PTO Damping [N/(m/s)]
pto(1).location = [0 0 0];                   % PTO Location [m]


%%  PTO-Sim Setup: 
%% Linear Generator PTO-Sim  
 
ptoSim(1) = ptoSimClass('PTOSim');
ptoSim(1).number  = 1;
ptoSim(1).type = 9; %Direct drive linear generator


%% Linear Generator

%params from doi: 10.1109/ECCE.2009.5316224.
% ptoSim(1).directLinearGenerator.Bfric = 0;%100;         % Friction coefficient
% ptoSim(1).directLinearGenerator.tau_p = 0.072;          % Magnet pole pitch [m]
% ptoSim(1).directLinearGenerator.lambda_fd = 8;          % Flux linkage of the stator d winding due to flux produced by the rotor magnets [Wb-turns]
% ptoSim(1).directLinearGenerator.lambda_sq_0 = 0;
% ptoSim(1).directLinearGenerator.lambda_sd_0 = ptoSim.directLinearGenerator.lambda_fd;  % (recognizing that the d-axis is always aligned with the rotor magnetic axis                        
% ptoSim(1).directLinearGenerator.Rs = 4.58;              % Winding resistance [ohm]
% ptoSim(1).directLinearGenerator.Ls = 0.285;             % Inductance of the coil [H], per-phase inductance *3/2
% ptoSim(1).directLinearGenerator.theta_d_0 = 0;

%params from doi: https://doi.org/10.1016/j.ecmx.2022.100190
%that got it from https://doi.org/10.1002/etep.56
ptoSim(1).directLinearGenerator.Bfric = 0;%100;         % Friction coefficient
ptoSim(1).directLinearGenerator.tau_p = 0.1;          % Magnet pole pitch [m]
ptoSim(1).directLinearGenerator.lambda_fd = 23;          % Flux linkage of the stator d winding due to flux produced by the rotor magnets [Wb-turns]
ptoSim(1).directLinearGenerator.lambda_sq_0 = 0;
ptoSim(1).directLinearGenerator.lambda_sd_0 = ptoSim.directLinearGenerator.lambda_fd;  % (recognizing that the d-axis is always aligned with the rotor magnetic axis                        
ptoSim(1).directLinearGenerator.Rs = 0.29;              % Winding resistance [ohm]
ptoSim(1).directLinearGenerator.Ls = 0.03;             % Inductance of the coil [H], per-phase inductance *3/2
ptoSim(1).directLinearGenerator.theta_d_0 = 0;

%TODO: Are there generator displacement limits? Is inertia important, or is
%the WEC body dominant?


%% Back-to-back converter parameters

%WEC-side converter
wsc.Bdamp = 100e3;%97e3; %Resistive damping coef
wsc.Kdamp = 0; %Reactive damping coef
wsc.Fpto_lim = pi*ptoSim(1).directLinearGenerator.lambda_fd^2/ptoSim(1).directLinearGenerator.Ls/ptoSim(1).directLinearGenerator.tau_p/2*0.999;
%NOTE: the Fpto limit is reduced to 99.9% to avoid a singularity in the
%simulation. Theoretically this shouldn't be needed but I think it happens
%due to small error accumulation from numerical calculations

%grid-side converter
gsc.Prated = 30e3;
gsc.Vmag = 480*1.1; %V, rms, l-l, 10% higher voltage than grid Vnom
gsc.Ilim = gsc.Prated/gsc.Vmag; %A, rms
gsc.Tavg = 5*60; %averaging period, s


%energy storage system
ess.Vdc_0 = gsc.Vmag*sqrt(2)*1.25; %V, 25% higher than grid voltage
ess.C = 88; %F
ess.Vdc_del = ess.Vdc_0-gsc.Vmag*sqrt(2); %max deviation from nominal voltage. When determining this value, ensure (Vdc_nom - Vdc_del) > gsc.Vmag*sqrt(2) ?





