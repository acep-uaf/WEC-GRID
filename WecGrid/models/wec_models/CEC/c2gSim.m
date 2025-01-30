function [m2g_out] = c2gSim(cecId, simLength)
    simLength = double(simLength);

    run('NewEnergy_20_ohms_100hz.m');
    r2g_ne5kW_init; % Call the function with simLength as an argument
    sim('R2G_ss_NE5kW.slx', [], simset('SrcWorkspace', 'current'));

    m2g_out.cecId = cecId;
end

