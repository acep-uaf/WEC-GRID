%% WEC-Sim to PSSe DATABASE data formatter
% Generate a SQLite file from the WEC-Sim Simulink simulation results. Must
% run or load simulation results first.

sim_data_struct = struct('time',0,'ibus',1,'machid','1','pg',[],'qg',[],'qt',[],'qb',[],'vs',1,...
    'ireg',0,'nreg',0,'mbase',[],'zr',[],'zx',[],'rt',[],'xt',[],'gtap',[],'stat',[],'rmpct',[],...
    'pt',[],'pb',[],'o1',[],'f1',[],'wmod',2,'wpf',0.5);

W2G_sample_size = length(logsout.getElement('Pgen').Values.Data);

W2G_data = repmat(sim_data_struct,W2G_sample_size-1,1); %minus 1 the sample size to ignore the Pgen during ramp up

%start at 2 to ignore the Pgen during ramp up
for i = 2:W2G_sample_size

    W2G_data(i).time = logsout.getElement('Pgen_pavg').Values.Time(i);
    W2G_data(i).pg = logsout.getElement('Pgen_pavg').Values.Data(i)/1e6; %in MW
    W2G_data(i).pt = logsout.getElement('Pgen_pavg').Values.Data(i)/1e6; %in MW
    W2G_data(i).pb = 0/1e6; %in MW
    W2G_data(i).qg = 0/1e6; %in Mvar
    W2G_data(i).qt = 50e3/1e6; %in Mvar
    W2G_data(i).qb = -50e3/1e6; %in Mvar
    

end
path = "../input_files/mysqlite.db";
dbfile = fullfile(path); 
conn = sqlite(dbfile,"create");

tablename = "WEC_output";
coltypes = ["numeric" "numeric" "numeric" "numeric" "varchar(255)"];

sqlwrite(conn, "WEC_output", struct2table(W2G_data))

close(conn)



