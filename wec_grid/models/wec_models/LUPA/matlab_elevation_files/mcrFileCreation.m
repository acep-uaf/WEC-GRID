clear
clc

% creating ETA files for MCR
% move all csv files to individual .mat files
% name all files with same naming convention 

Tp = 11; % sec
Tp_str = '11_0';

Hs = 3; % m
Hs_str = '3_0';

mcr_name = 'mcrEta_TEST.mat';

WaveNum = [20,50]; %,100,250,500];

percentile ={'05','25','50','75','95'};

header = {'waves.elevationFile','simu.endTime'};

mcr.header = {'waves.elevationFile','simu.endTime'};

count = 0;

for i = 1:length(WaveNum)
    fn = [num2str(WaveNum(i)),'WaveEta.csv'];

    C = readcell(fn)';
    S = string(C(1,:));
    S(1,1) = 'Time';
    T = cell2table(C(2:end,:),'VariableNames',S);
    
    for j = 1:length(percentile)
        count = count + 1; 
        case_names{count} = ['ETAs\',num2str(WaveNum(i)),'waveEta_', ...
        percentile{j},'percentile_',Hs_str,'_Hs_' Tp_str,'_Tp.mat'];

        % saving data to file

        eta(:,1) = T(:,1);
        eta(:,2) = T(:,j+1);

        eta = table2array(eta);

        %%% comment out lines 47-55 for normalized files
        eta(:,1) = eta(:,1)*Tp;
        eta(:,2) = eta(:,2)*Hs;

        xq = 0:0.04:eta(end,1);

        new_eta(:,2) = interp1(eta(:,1),eta(:,2),xq,'spline')';
        new_eta(:,1) = xq';

        eta = new_eta;
%     
        save(case_names{count},'eta')

        % Pulling end time 
        endtimes(count) = eta(end,1);

        clear eta new_eta

    end

end


case_names = case_names';
endtimes = endtimes';

% Generating MRC .mat file

mcr.cases = table2cell(table(case_names(:),endtimes(:)));

save(mcr_name,'mcr')
