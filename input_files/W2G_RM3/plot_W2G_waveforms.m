
% Pgen = m2g_out.Pgen;
% Pgrid_ds = m2g_out.Pgrid;
% Qgrid_lim_ds = m2g_out.Qgrid_lim;

tend = m2g_out.Pgrid.Time(end);
Ts = gsc.Tavg;

Pgrid_ds = DownSampleTS(m2g_out.Pgrid,Ts,1);
Qgrid_lim_ds = DownSampleTS(m2g_out.Qgrid_lim,Ts,1);

%% Fig 1
figure(1)
t1 = tiledlayout(3,1);
t1.Padding = 'compact';
t1.TileSpacing = 'tight';


nexttile
plot(m2g_out.Pgen.Time/60,m2g_out.Pgen.Data/1e3)
% legend('WEC')
grid on
xticks(0:Ts:tend)
xlim([0 tend])
xticks(0:Ts/60:tend/60)
xlim([0 tend/60])
ylabel('kW')
title('WEC Generated Power')

nexttile
plot(Pgrid_ds.Time/60,Pgrid_ds.Data/1e3,'.-','LineWidth',1,'MarkerSize',15)
hold on
plot(m2g_out.Pgrid.Time/60,m2g_out.Pgrid.Data/1e3)
hold off
legend('Static','Dynamic','Location','southeast')
grid on
xticks(0:Ts/60:tend/60)
xlim([0 tend/60])
ylabel('kW')
title('Grid Active Power')

nexttile
plot(Qgrid_lim_ds.Time/60,Qgrid_lim_ds.Data/1e3,'.-','LineWidth',1,'MarkerSize',15)
hold on
plot(m2g_out.Qgrid_lim.Time/60,m2g_out.Qgrid_lim.Data/1e3)
hold off
legend('Static','Dynamic','Location','southeast')
grid on
xticks(0:Ts/60:tend/60)
xlim([0 tend/60])
% yticks(0:5:65)
ylabel('kVAr')
title('Maximum Grid Reactive Power')
xlabel('Time (min)')


% f1 = figure(1);
% f1.Units = 'Inches'; 
% f1.Position = [1, 1, 13, 6];
% image(breck);
% filename = 'Breckenridge';
% print(f1, filename, '-djpeg', '-r300');


%% Fig 2
figure(2)
t2 = tiledlayout(1,1);
t2.Padding = 'compact';
t2.TileSpacing = 'tight';


nexttile
plot(m2g_out.Pgen.Time/60,m2g_out.Pgen.Data/1e3,'Color',0.6*[1 1 1])
hold on
plot(m2g_out.Pgrid.Time/60,m2g_out.Pgrid.Data/1e3,'LineWidth',2)
hold off
legend('Instantaneous Power','Power Smoothing')
grid on
% xticks(0:Ts:tend)
% xlim([0 tend])
xticks(0:Ts/60:tend/60)
xlim([0 tend/60])
xlabel('Time (min)')
ylabel('kW')
ylim([0 80])
yticks([0:10:80])
title('WEC Generated Electrical Power')