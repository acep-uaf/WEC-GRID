
simu.cicEndTime = 20; %default is 60 seconds, doesn't work for the current model version. 20 does.

%% Body Data
%% Body 1: Float
body(1) = bodyClass('.\model_library\wecs\LUPA\hydroData\lupa.h5');
body(1).geometryFile = '.\model_library\wecs\LUPA\geometry\LUPA_Fall2022_float_geometry.stl';
body(1).mass = 'equilibrium';
body(1).inertia = [66.1686 65.3344 17.16];    %[kg-m^2]

%% Body 2: Spar
body(2) = bodyClass('.\model_library\wecs\LUPA\hydroData\lupa.h5');
body(2).geometryFile = '.\model_library\wecs\LUPA\geometry\LUPA_Fall2022_spar_geometry.stl';
body(2).mass = 'equilibrium';
body(2).inertia = [253.6344 250.4558 12.746];    %[kg-m^2]