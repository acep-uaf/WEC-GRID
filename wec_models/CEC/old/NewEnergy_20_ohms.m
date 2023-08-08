clc, clear 

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
dt_ADCP= diff(time_ADCP);
dt_ADCP.Format = 's'; 
dt_ADCP_avg=mean(dt_ADCP);

ADCP=timetable(velocity_ADCP(:,1,1),'RowTimes',time_ADCP );
ADCP = retime(ADCP, 'regular', 'linear', 'TimeStep',dt_ADCP_avg);

ADCP_vel = [seconds(ADCP.Time-ADCP.Time(1)),  ADCP.Var1];

%nccreate( '5KW_sim.nc','out.simout')

% run model here


% electircal | ts 

% ts @ .1 seconds (need o)