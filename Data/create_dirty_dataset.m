% Create dataset with only 100 pts from -10ms to 90ms
% to match old dataset

clear
close all

load('data_fall2017.mat')
load('data_spring2018.mat')

n_fall = length(data_fall2017);
n_spring = length(data_spring2018);
data = zeros(n_fall+n_spring,199,6);

% % trimmed dataset
% for i=1:n_fall
%     t_start(i) = find(data_fall2017(i).t>=-.01,1);
%     data(i,:,:) = [data_fall2017(i).lin_acc_CG(t_start(i):t_start(i)+99,:),data_fall2017(i).ang_vel(t_start(i):t_start(i)+99,:)];
% end
% 
% for i=1:n_spring
%     t_start(i+n_fall) = find(data_spring2018(i).t>=-.01,1);
%     data(i+n_fall,:,:) = [data_spring2018(i).lin_acc_CG(t_start(i):t_start(i)+99,:),data_spring2018(i).ang_vel(t_start(i):t_start(i)+99,:)];
%     
% end

% full dataset (all points)
for i=1:n_fall
    data(i,:,:) = [data_fall2017(i).lin_acc_CG(1:199,:),data_fall2017(i).ang_vel(1:199,:)];
end

for i=1:n_spring
    data(i+n_fall,:,:) = [data_spring2018(i).lin_acc_CG(1:199,:),data_spring2018(i).ang_vel(1:199,:)];
end

labels = [labels_fall2017;labels_spring2018];

% randomize data and labels
index = randperm(n_fall+n_spring);

data_dirty = data(index,:,:);
labels_dirty = labels(index);

save('data_dirty.mat','data_dirty')
save('labels_dirty.mat','labels_dirty')