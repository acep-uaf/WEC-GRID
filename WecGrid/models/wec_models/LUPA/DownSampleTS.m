function [Xds] = DownSampleTS(X,T,timeshift)
%Downsample a time-series given the new sample period.
% NOTE: Assuming a fixed time step 
% timeshift: set to 1 for the samples to be centered on the period, 0 for
% the samples to be at end of period

Told = X.Time(2) - X.Time(1);
Tsample = floor(T/Told);
NewSampleSize = floor((X.Time(end) - X.Time(1))/T);

%pre-allocate new timeseries, 
if timeshift == 1
    % time-shift so that new timeseries points are
    %centered within the averaged frames
    Xds = timeseries(zeros(NewSampleSize,1),(T/2):T:(NewSampleSize*(T-1)+T/2));
else
    Xds = timeseries(zeros(NewSampleSize,1),T:T:(NewSampleSize*T));
end


for i = 1:NewSampleSize %X.Time(1):T:X.Time(end)
    if i == 1
        Xds.Data(i) = mean(X.Data(1:i*Tsample));
    else
        Xds.Data(i) = mean(X.Data((i-1)*Tsample:i*Tsample));
    end
    
end

end

