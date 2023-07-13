Pr=6e3; % Rated Power
Nr= 150; % Rated Speed in RPM
Wr=2*pi*Nr/60;
Tr=Pr/Wr;
Rs=3.711; % Stator Resistance
Lsd=45.25e-3; % Stator d-axis Inductance
Lsq=64.88e-3; % Stator q-axis Inductance
P=24; % Number of poles
Eb=440; % back emf
k= Eb/(150*2*pi/60)/sqrt(3) % machine constant (Vs/rad)
lamda_fd=3*2*k/P; % Flux Constant
J=0.2564; % Inertia Constant kg/m^2
%Assume 1% friction loss 
B=0.01*Pr/(Wr^2); % Friction Co-efficient