%% pto generator params
run('.\init_pto_gen.m')


%% Back-to-back converter parameters
%grid-side converter
gsc.Prated = 1e3;
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

