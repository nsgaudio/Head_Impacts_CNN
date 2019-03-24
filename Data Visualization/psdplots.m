load('training_data_fall2017.mat')
load('labels_fall2017.mat')
impacts = find(label_impact_noimpact == 1);
noimpacts = find(label_impact_noimpact == 0);

T = 0.0001;
Fs = 1/T;

% %impact ang vel x
% figure; hold on;
% for i=1:length(impacts)
%     [f,y] = fft_freq(training_data(impacts(i)).ang_vel(:,1), Fs);
%     plot(f,y);
% end
% 
% %impact ang vel y
% figure; hold on;
% for i=1:length(impacts)
%     [f,y] = fft_freq(training_data(impacts(i)).ang_vel(:,2), Fs);
%     plot(f,y);
% end
% 
% %impact ang vel z
% figure; hold on;
% for i=1:length(impacts)
%     [f,y] = fft_freq(training_data(impacts(i)).ang_vel(:,3), Fs);
%     plot(f,y);
% end
% 
% %impact ang vel mag
% figure; hold on;
% for i=1:length(impacts)
%     [f,y] = fft_freq(training_data(impacts(i)).ang_vel_mag, Fs);
%     plot(f,y);
% end

%angular velocity impact
figure
hold on
[fx_impact,yx_impact] = fft_freq(training_data(impacts(5)).ang_vel(:,1), Fs);
[fy_impact,yy_impact] = fft_freq(training_data(impacts(5)).ang_vel(:,2), Fs);
[fz_impact,yz_impact] = fft_freq(training_data(impacts(5)).ang_vel(:,3), Fs);
[fmag_impact,ymag_impact] = fft_freq(training_data(impacts(5)).ang_vel_mag, Fs);
plot(fx_impact,yx_impact,'linewidth',2)
plot(fy_impact, yy_impact,'linewidth',2)
plot(fz_impact, yz_impact,'linewidth',2)
plot(fmag_impact, ymag_impact,'k','linewidth',2)
legend('x','y','z','mag');
title('ang vel impact')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
set(gca,'fontsize',15)

%linear acceleration impact
figure
hold on
[fx_impact,yx_impact] = fft_freq(training_data(impacts(5)).lin_acc_CG(:,1), Fs);
[fy_impact,yy_impact] = fft_freq(training_data(impacts(5)).lin_acc_CG(:,2), Fs);
[fz_impact,yz_impact] = fft_freq(training_data(impacts(5)).lin_acc_CG(:,3), Fs);
[fmag_impact,ymag_impact] = fft_freq(training_data(impacts(5)).lin_acc_CG_mag, Fs);
plot(fx_impact,yx_impact,'linewidth',2)
plot(fy_impact, yy_impact,'linewidth',2)
plot(fz_impact, yz_impact,'linewidth',2)
plot(fmag_impact, ymag_impact,'k','linewidth',2)
legend('x','y','z','mag');
title('lin acc impact')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
set(gca,'fontsize',15)


% %no impact ang vel mag
% figure; hold on;
% for i=1:length(noimpacts)
%     [f,y] = fft_freq(training_data(noimpacts(i)).ang_vel_mag, Fs);
%     plot(f,y);
% end

%angular velocity no impact
figure
hold on
[fx_noimpact,yx_noimpact] = fft_freq(training_data(noimpacts(16)).ang_vel(:,1), Fs);
[fy_noimpact,yy_noimpact] = fft_freq(training_data(noimpacts(16)).ang_vel(:,2), Fs);
[fz_noimpact,yz_noimpact] = fft_freq(training_data(noimpacts(16)).ang_vel(:,3), Fs);
[fmag_noimpact,ymag_noimpact] = fft_freq(training_data(noimpacts(16)).ang_vel_mag, Fs);
plot(fx_noimpact,yx_noimpact,'linewidth',2)
plot(fy_noimpact, yy_noimpact,'linewidth',2)
plot(fz_noimpact, yz_noimpact,'linewidth',2)
plot(fmag_noimpact, ymag_noimpact,'k','linewidth',2);
legend('x','y','z','mag');
title('ang vel no impact');
xlabel('Frequency (Hz)')
ylabel('Amplitude')
set(gca,'fontsize',15)

%linear acceleration no impact
figure
hold on
[fx_impact,yx_impact] = fft_freq(training_data(noimpacts(16)).lin_acc_CG(:,1), Fs);
[fy_impact,yy_impact] = fft_freq(training_data(noimpacts(16)).lin_acc_CG(:,2), Fs);
[fz_impact,yz_impact] = fft_freq(training_data(noimpacts(16)).lin_acc_CG(:,3), Fs);
[fmag_impact,ymag_impact] = fft_freq(training_data(noimpacts(16)).lin_acc_CG_mag, Fs);
plot(fx_noimpact,yx_noimpact,'linewidth',2)
plot(fy_noimpact, yy_noimpact,'linewidth',2)
plot(fz_noimpact, yz_noimpact,'linewidth',2)
plot(fmag_noimpact, ymag_noimpact,'k','linewidth',2);
legend('x','y','z','mag');
title('lin acc no impact');
xlabel('Frequency (Hz)')
ylabel('Amplitude')
set(gca,'fontsize',15)

