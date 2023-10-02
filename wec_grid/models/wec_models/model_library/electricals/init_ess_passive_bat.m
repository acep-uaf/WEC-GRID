function [ess] = init_ess_passive_bat(ess)
%configure energy storage model to behave as battery
ess.isActive = 0;

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


end

