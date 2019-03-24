% process Lyndia's datset for use in Neural net

clear
close all


load('Z:\Data\TBI\ImpactDetection\data_impactsNonimpacts_11-11-2016.mat')


%%
training_data = [impacts_hiIR_dirMatch_notNoise nonimpacts_hiIR];

% The output vector y
labels = [ones(n_impacts_hiIR_dirMatch_notNoise,1); zeros(n_nonimpacts_hiIR,1)];

n = length(training_data);
data = zeros(n,100,6);
t_start = zeros(1,n);
for i=1:n
    data(i,:,:) = [training_data(i).lin_acc_CG,training_data(i).ang_vel];
end

% randomize data and labels
index = randperm(n);

labels_lyndia = labels(index);
data_lyndia = data(index,:,:);

save('data_lyndia.mat','data_lyndia')
save('labels_lyndia.mat','labels_lyndia')