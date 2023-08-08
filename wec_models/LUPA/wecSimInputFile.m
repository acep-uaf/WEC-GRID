%% LUPA Two Body Heave Only
% Made by Courtney Beringer
% August 2022, edited June 2023
%{
Inputs
    Flume width and water depth:     Measured from experiments in Fall
                                     2022.
    H5 files:                        Float and Spar body geometries were
                                     meshed, run through WAMIT, and
                                     WEC-Sim's BEMIO.
    Momements of inertia:            Calculated experimentally from swing
                                     tests in Fall 2022.
    Mass:                            Set to equilibrium as the WAMIT data
                                     has the center of mass at the origin.
    PTO damping and stiffness:       In the experiments, LUPA PTO damping
                                     varied between 0-7000 N/m/s, but
                                     higher is of interest as well.
                                     Stiffness was also experimented with.
    Waves:                           Left blank for user input. Wave period
                                     varied from 1.25-5 seconds. Wave
                                     height varied from 0.1-0.2 meters in
                                     experiments for regular waves. LUPA is
                                     1:25 scale.
    Constraint:                      This example is for the two body heave 
                                     only configuration of LUPA. The 
                                     simulink file constrains the spar to
                                     heave motions.
%}

%% Simulation Data #TODO: split up and organize the input files to be wec and rest of w2g specific
simu=simulationClass();
% simu.simMechanicsFile = 'LUPAsim.slx';
% simu.rampTime = 10;
% simu.endTime = 60;                        
% simu.dt = 0.01;
% simu.explorer = 'off';
% halfFlume = 3.7/2;
% simu.domainSize = halfFlume;   % Width of the flume is 3.7 meters. The domain size needs half of that. 
%output.saveViz(simu,body,waves,'timesPerFrame',5,'axisLimits',[-halfFlume
%halfFlume -halfFlume halfFlume -2.7 2]); %If you want to wave the video

%% Wave Information  
% No Wave
% waves = waveClass('noWaveCIC');                   % Initialize waveClass

% waves = waveClass('regular');
% waves.height = 0.2; %[m]
% waves.period = 2; %[s]
%% Body Data
%% Body 1: Float
body(1) = bodyClass('./hydroData\lupa.h5');
body(1).geometryFile = './geometry\LUPA_Fall2022_float_geometry.stl';
body(1).mass = 'equilibrium';
body(1).inertia = [66.1686 65.3344 17.16];    %[kg-m^2]

%% Body 2: Spar
body(2) = bodyClass('./hydroData\lupa.h5');
body(2).geometryFile = './geometry\LUPA_Fall2022_spar_geometry.stl';
body(2).mass = 'equilibrium';
body(2).inertia = [253.6344 250.4558 12.746];    %[kg-m^2]

%% PTO and Constraint Parameters
% Translational
constraint(1) = constraintClass('Constraint1'); % Initialize Constraint Class for Constraint1
constraint(1).location = [0 0 0];                    % Constraint Location [m]

% Translational PTO
% pto(1) = ptoClass('PTO1');                      % Initialize PTO Class for PTO1
% pto(1).stiffness = 0;                                   % PTO Stiffness [N/m]
% pto(1).damping = 0;                                   % PTO Damping [N/(m/s)]
% pto(1).location = [0 0 0];                           % PTO Location [m]


%% MUST ADD
run("./W2GInputFile.m")
