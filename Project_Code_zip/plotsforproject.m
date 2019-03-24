clear all
close all
clc
load('training_data_fall2017.mat')
load('labels_fall2017.mat')
load('data.mat')

%split into postitive and negative labels
impacts = find(label_impact_noimpact == 1);
noimpacts = find(label_impact_noimpact == 0);

%impact plots
%plot linacc x
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).lin_acc_CG(:,1))
end

%plot linacc y
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).lin_acc_CG(:,2))
end

%plot linacc z
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).lin_acc_CG(:,3))
end

%plot linacc mag
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).lin_acc_CG_mag)
end




%plot ang vel x
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).ang_vel(:,1))
end

%plot ang vel y
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).ang_vel(:,2))
end

%plot ang vel z
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).ang_vel(:,3))
end

%plot ang vel mag
figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).ang_vel_mag)
end



%no impact plots
%plot linacc x
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).lin_acc_CG(:,1))
end

%plot linacc y
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).lin_acc_CG(:,3))
end

%plot linacc z
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).lin_acc_CG(:,2))
end

%plot linacc mag
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).lin_acc_CG_mag)
end




%plot ang vel x
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).ang_vel(:,1))
end

%plot ang vel y
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).ang_vel(:,2))
end

%plot ang vel z
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).ang_vel(:,3))
end

%plot ang vel mag
figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).ang_vel_mag)
end

%%
close all

figure
hold on
t = training_data(noimpacts(16)).t;
vx = training_data(noimpacts(16)).ang_vel(:,1);
vy = training_data(noimpacts(16)).ang_vel(:,2);
vz = training_data(noimpacts(16)).ang_vel(:,3);
vmag = training_data(noimpacts(16)).ang_vel_mag;
plot(t,vx,'linewidth',2)
plot(t,vy,'linewidth',2)
plot(t,vz,'linewidth',2)
plot(t,vmag,'k','linewidth',2);
xlim([-.01,0.1])
legend('x','y','z','mag');
%title('No Impact angular velocity vs. time for one example');
set(gca,'fontsize',15)
xlabel('Time')
ylabel('rad/s')

figure
hold on
t = training_data(impacts(5)).t;
lx = training_data(impacts(5)).lin_acc_CG(:,1);
ly = training_data(impacts(5)).lin_acc_CG(:,2);
lz = training_data(impacts(5)).lin_acc_CG(:,3);
lmag = training_data(impacts(5)).lin_acc_CG_mag;
plot(t,lx,'linewidth',2)
plot(t,ly,'linewidth',2)
plot(t,lz,'linewidth',2)
plot(t,lmag,'k','linewidth',2);
xlim([-.01,0.1])
legend('x','y','z','mag');
%title('Impact linear acceleration vs. time for one example');
set(gca,'fontsize',15)
xlabel('Time')
ylabel('g')

figure
hold on
t = training_data(noimpacts(16)).t;
lx = training_data(noimpacts(16)).lin_acc_CG(:,1);
ly = training_data(noimpacts(16)).lin_acc_CG(:,2);
lz = training_data(noimpacts(16)).lin_acc_CG(:,3);
lmag = training_data(noimpacts(16)).lin_acc_CG_mag;
plot(t,lx,'linewidth',2)
plot(t,ly,'linewidth',2)
plot(t,lz,'linewidth',2)
plot(t,lmag,'k','linewidth',2);
xlim([-.01,0.1])
legend('x','y','z','mag');
%title('No Impact linear acceleration vs. time for one example');
set(gca,'fontsize',15)
xlabel('Time')
ylabel('g')



figure
hold on
vx = training_data(impacts(5)).ang_vel(:,1);
vy = training_data(impacts(5)).ang_vel(:,2);
vz = training_data(impacts(5)).ang_vel(:,3);
vmag = training_data(impacts(5)).ang_vel_mag;
plot(t,vx,'linewidth',2)
plot(t,vy,'linewidth',2)
plot(t,vz,'linewidth',2)
plot(t,vmag,'k','linewidth',2);
xlim([-.01,0.1])
legend('x','y','z','mag');
%title('Impact angular velocity vs. time for one example');
set(gca,'fontsize',15)
xlabel('Time')
ylabel('rad/s')




