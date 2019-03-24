%

clear
close all

load('training_data_fall2017.mat')
load('labels_fall2017');

impacts = find(label_impact_noimpact == 1);
noimpacts = find(label_impact_noimpact == 0);

figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).lin_acc_CG_mag)
end

figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).lin_acc_CG_mag)
end


figure
hold on
for i = 1:length(impacts)
    plot(training_data(impacts(i)).t,training_data(impacts(i)).ang_vel_mag)
end

figure
hold on
for i = 1:length(noimpacts)
    plot(training_data(noimpacts(i)).t,training_data(noimpacts(i)).ang_vel_mag)
end