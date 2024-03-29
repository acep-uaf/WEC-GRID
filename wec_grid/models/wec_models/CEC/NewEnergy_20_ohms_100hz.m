
filenameADCP='ds_streamwise_20ohms.nc';
velocity_ADCP= ncread(filenameADCP, 'vel');
time_ADCP_raw= ncread(filenameADCP, 'time');
range_ADCP_raw= ncread(filenameADCP, 'range');

    IT= ncreadatt(filenameADCP,"time","units");
    
    units_ADCP= split(IT);
    IT_AD2CP=datetime(append(units_ADCP(3), ' ',units_ADCP(4)), 'InputFormat','yyyy-MM-dd HH:mm:ss.SSSSSS');
    time_ADCP = IT_AD2CP+seconds(double(time_ADCP_raw)/1e9);
    time_ADCP.Format = 'HH:mm:ss.SSSSSS'; 

% dt for ADCP 
dt= seconds(0.1);

ADCP=timetable(velocity_ADCP(:,1,1),'RowTimes',time_ADCP );
ADCP = retime(ADCP, 'regular', 'previous', 'TimeStep',dt);

ADCP_vel = [seconds(ADCP.Time-ADCP.Time(1)),  ADCP.Var1];
ADCP_vel = rmmissing(ADCP_vel );
%nccreate( '5KW_sim.nc','out.simout')

ADCP_vel_struct.time = ADCP_vel(:,1);
ADCP_vel_struct.signals.values= ADCP_vel(:,2);

%%
