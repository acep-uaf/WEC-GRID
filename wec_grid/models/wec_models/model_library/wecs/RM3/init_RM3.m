%% Body Data
% Float
body(1) = bodyClass('.\model_library\wecs\RM3\hydroData\rm3.h5');             
body(1).geometryFile = '.\model_library\wecs\RM3\geometry\float.stl';      
body(1).mass = 'equilibrium';                   
body(1).inertia = [20907301 21306090.66 37085481.11];     
body(1).quadDrag.cd = ([1 1 1 0 0 0]);
body(1).quadDrag.area = ([5*20 5*20 (14^2)*pi 0 0 0]);


% Spar/Plate
body(2) = bodyClass('.\model_library\wecs\RM3\hydroData\rm3.h5');     
body(2).geometryFile = '.\model_library\wecs\RM3\geometry\plate.stl';  
body(2).mass = 'equilibrium';                   
body(2).inertia = [94419614.57 94407091.24 28542224.82];
body(2).quadDrag.cd = ([2.56 2.56 3.21 0 0 0]);
body(2).quadDrag.area = ([(0.1*30+6*38) (0.1*30+6*38) (15^2)*pi 0 0 0]);


