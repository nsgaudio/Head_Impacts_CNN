% Create dataset with only 100 pts from -10ms to 90ms
% to match old dataset

clear
close all

load('training_data_fall2017.mat')
n = length(training_data);
data = zeros(n,100,6);
t_start = zeros(1,n);
for i=1:n
    t_start(i) = find(training_data(i).t>=-.01,1);
    data(i,:,:) = [training_data(i).lin_acc_CG(t_start(i):t_start(i)+99,:),training_data(i).ang_vel(t_start(i):t_start(i)+99,:)];
end

data_trimmed = data;

save('data_trimmed.mat','data_trimmed')