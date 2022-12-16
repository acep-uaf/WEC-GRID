%% WEC-Sim to PSSe data formatter
% Generate a csv file from the WEC-Sim Simulink simulation results. Must
% run or load simulation results first.

%Refer to PSSe Data Format Documentation, S1.10 Generator Data (pg. 15)
%Data to be populated by W2G sim:
% PG: set to Pgen_avg
% QG: set to 0
% QT: determine max Q for ACDC inverter
% QB: determine min ""
% VS: set to 1
% PT: set to Pgen_avg
% PB: set to 0? Or determine a minimum? For dynamic PF it won't be 0
% WMOD: not directly from W2G sim, but need to determine this
% WPF: Might need this for WMOD

sim_data_struct = struct('time',0,'ibus',1,'machid','1','pg',0,'qg',0,'qt',0,'qb',0,'vs',1,...
    'ireg',0,'nreg',0,'mbase',[],'zr',[],'zx',[],'rt',[],'xt',[],'gtap',[],'stat',[],'rmpct',[],...
    'pt',0,'pb',0,'o1',[],'f1',[],'wmod',2,'wpf',0.5);

%downsample the simulated data
Pgrid_ds = DownSampleTS(m2g_out.Pgrid,gsc.Tavg,1);
Qgrid_lim_ds = DownSampleTS(m2g_out.Qgrid_lim,gsc.Tavg,1);

%determine number of entries in output file
W2G_sample_size = length(Pgrid_ds.Data);

% W2G_dt = simu.dt;

% W2G_data = repmat(sim_data_struct,W2G_sample_size-1,1); %minus 1 the sample size to ignore the Pgen during ramp up
W2G_data = repmat(sim_data_struct,W2G_sample_size,1); %minus 1 the sample size to ignore the Pgen during ramp up


%start at 2 to ignore the Pgen during ramp up
for i = 2:W2G_sample_size

    W2G_data(i).time = Pgrid_ds.Time(i);
    W2G_data(i).pg = Pgrid_ds.Data(i)/1e3; %in kW
    W2G_data(i).pt = Pgrid_ds.Data(i)/1e3; %in kW
    W2G_data(i).pb = 0/1e3; %in kW
    W2G_data(i).qg = 0/1e3; %in kvar
    W2G_data(i).qt = Qgrid_lim_ds.Data(i)/1e3; %in kvar
    W2G_data(i).qb = -Qgrid_lim_ds.Data(i)/1e3; %in kvar

end

%writetable(struct2table(W2G_data),'../genData.csv','Delimiter','comma')

%path = "./mysqlite.db";
%dbfile = fullfile(path); 
%conn = database("compare_sales.sql","","");
%conn = database('mysql.db');

%tablename = "WEC_output";
%sqlquery = ["CREATE TABLE WEC_output( time numeric, ibus numeric, pt numeric, pb numeric, qg numeric,  qt numeric, qb numeric)"];
%exec(conn, sqlquery)
 
data = struct2table(W2G_data);
data =  data(:,["time" "ibus" "pg" "pt" "pb" "qg" "qt" "qb"]);
writetable(data,'../genData.csv','Delimiter','comma')

%colnames = data.Properties.VariableNames;
%colnames = ["time","ibus","pt","pb","qg","qt","qb"];

%insert(conn, tablename, colnames, data);

%close(conn)

