function [m2g_out] = CEC2G_sim(cecId, simLength)
    simLength = double(simLength);

    run('.\model_library\cecs\NE5kW\NewEnergy_20_ohms_100hz.m');
    run('.\model_library\cecs\NE5kW\r2g_ne5kW_init.m')%r2g_ne5kW_init; % Call the function with simLength as an argument
    sim('CEC2G_ss.slx', [], simset('SrcWorkspace', 'current'));

    m2g_out.cecId = cecId;
end

