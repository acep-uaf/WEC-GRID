function [ess] = ess_passive_sc_config(ess)
%configure energy storage model to behave as a capacitor or supercapicitor


%determine max and min capacitor pack voltage levels
%set max voltage of sc so that nominal voltage (Vdc_0) equates to 50% of
%energy stored
ess.Vmax = ess.Vdc_0/sqrt(0.5);
ess.Vmin = 0;

ess.SOC_LUT = 0:0.01:1; %range of SOC values in lookup table

%assume a normal capacitor voltage curve
ess.Voc_LUT = ess.Vmax*ess.SOC_LUT; 


ess.Csoc = ess.Ecap*3600*2/ess.Vmax; %SOC capacitance, A-s
% in an electrical equivalent circuit, voltage across this "capacitor" is 
% equivalent to the SOC.

ess.SOC_0 = ess.Vdc_0/ess.Vmax; %initial SOC of sc, set to Vdc_nom

ess.eff = 1.0;%0.90; %efficiency of sc

end

