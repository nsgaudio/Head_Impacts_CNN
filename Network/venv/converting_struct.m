clear all
close all
clc
load('training_data_fall2017.mat');

    

data = zeros(length(training_data),length(training_data(1).t),6);

for i=1:length(training_data)
    if length(training_data(i).t) == 199
        data(i,:,1:3) = training_data(i).lin_acc_CG;
        data(i,:,4:6) = training_data(i).ang_vel;
    elseif length(training_data(i).t) > 199
        data(i,:,1:3) = training_data(i).lin_acc_CG(1:199,:);
        data(i,:,4:6) = training_data(i).ang_vel(1:199,:);
    else
        error('asdjfl;asdj;flja;')
    end
end
    



