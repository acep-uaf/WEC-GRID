%% Simulation Data
simu = simulationClass();        
simu.simMechanicsFile = './WEC2G_ss.slx';      %Location of Simulink Model File with PTO-Sim  
simu.explorer = 'off';                  % Turn SimMechanics Explorer (on/off)
simu.mode = 'normal';                   % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime = 0;                      % Wave Ramp Time [s]
simu.endTime = simLength;                     % Simulation End Time [s]
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
simu.dt = 0.1; 	

%% Wave Information
% Regular Waves  
% waves = waveClass('regular');            
% waves.height = 2.5;                          
% waves.period = 8;  

%Irregular Waves using PM Spectrum
%waves = waveClass('elevationImport');
waves = waveClass('irregular');
%waves.elevationFile = 'GeneratedEtaCorrected.mat';
waves.height = waveHeight;
waves.period = wavePeriod;
waves.spectrumType = 'PM';
waves.phaseSeed = waveSeed;

%The equal energy formulation speeds up the irregular wave simulation time
% by reducing the number of frequencies the wave train is defined by,
% and thus the number of frequencies for which the wave forces are
% calculated. It prevents bins with very little energy from being created
% and unnecessarily adding to the computational cost.
% waves.bem.option = 'EqualEnergy';


% run('.\model_library\wecs\RM3\init_RM3.m')
% run('.\model_library\wecs\RM3\init_w2w_system.m')

run('.\model_library\wecs\LUPA\init_LUPA.m')
run('.\model_library\wecs\LUPA\init_w2w_system.m')

