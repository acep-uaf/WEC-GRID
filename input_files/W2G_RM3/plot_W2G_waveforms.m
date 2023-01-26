%% Script to plot waveforms after simulation

%get simulation length
tend = m2g_out.Pgrid.Time(end);


%% Fig 1
f1 = figure(1);
f1.Units = 'inches'; 
f1.Position = [1, 1, 6, 4];
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
title('Resource Generated Power')

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
title('Grid Reactive Power Limit')
xlabel('Time (min)')


%% Fig 2
f2 = figure(2);
f2.Units = 'inches'; 
f2.Position = [2, 1, 6, 3];
t2 = tiledlayout(1,1);
t2.Padding = 'compact';
t2.TileSpacing = 'tight';

nexttile
plot(m2g_out.Pgen.Time/60,m2g_out.Pgen.Data/1e3,'Color',0.6*[1 1 1])
hold on
plot(m2g_out.Pgrid.Time/60,m2g_out.Pgrid.Data/1e3,'LineWidth',2)
hold off
legend('Resource Generated Power','Grid Delivered Power')
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


%% Fig 3
f3 = figure(3);
f3.Units = 'inches'; 
f3.Position = [3, 1, 6, 3];
t3 = tiledlayout(1,1);
t3.Padding = 'compact';
t3.TileSpacing = 'tight';

nexttile
plot(m2g_out.Vdc.Time/60,m2g_out.Vdc.Data)
hold on
yline(ess.Vdc_0 + ess.Vdc_del, '--','Label','Upper Limit')
yline(ess.Vdc_0 - ess.Vdc_del, '--','Label','Lower Limit')
hold off
grid on
xticks(0:Ts:tend)
xlim([0 tend])
xticks(0:Ts/60:tend/60)
xlim([0 tend/60])
ylabel('V')
title('DC Link Voltage')


%% Save images

print(f1, './sim_figures/Pgen_Pgrid_Qgrid', '-djpeg', '-r350');
print(f2, './sim_figures/Pgen_Pgrid_comp', '-djpeg', '-r350');
print(f3, './sim_figures/DClink_voltage', '-djpeg', '-r350');


