%% PTO and Constraint Parameters
% Translational Constraint
constraint(1) = constraintClass('Constraint1'); 
constraint(1).location = [0 0 0]; 

% Translational PTO
pto(1) = ptoClass('PTO1');           	% Initialize PTO Class for PTO1
pto(1).stiffness = 0;                           % PTO Stiffness [N/m]
pto(1).damping = 0;                           % PTO Damping [N/(m/s)]
pto(1).location = [0 0 0];                   % PTO Location [m]

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


%% PTO controlerl params
wsc.Bdamp = 100e3;%97e3; %Resistive damping coef
wsc.Kdamp = 0; %Reactive damping coef
wsc.Fpto_lim = pi*ptoSim(1).directLinearGenerator.lambda_fd^2/ptoSim(1).directLinearGenerator.Ls/ptoSim(1).directLinearGenerator.tau_p/2*0.999;
%NOTE: the Fpto limit is reduced to 99.9% to avoid a singularity in the
%simulation. Theoretically this shouldn't be needed but I think it happens
%due to small error accumulation from numerical calculations

