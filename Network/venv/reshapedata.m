clear all
close all
clc
load('data.mat')

lin_acc_x = data(:,:,1);
lin_acc_y = data(:,:,2);
lin_acc_z = data(:,:,3);
ang_vel_x = data(:,:,4);
ang_vel_y = data(:,:,5);
ang_vel_z = data(:,:,6);


mat2np(ang_vel_x,'ang_vel_x.pkl','float64')
mat2np(ang_vel_y,'ang_vel_y.pkl','float64')
mat2np(ang_vel_z,'ang_vel_z.pkl','float64')
mat2np(lin_acc_x,'lin_acc_x.pkl','float64')
mat2np(lin_acc_y,'lin_acc_y.pkl','float64')
mat2np(lin_acc_z,'lin_acc_z.pkl','float64')
mat2np(label_impact_noimpact,'labels.pkl','float64')



csvwrite('ang_vel_x.csv',ang_vel_x)
csvwrite('ang_vel_y.csv',ang_vel_y)
csvwrite('ang_vel_z.csv',ang_vel_z)
csvwrite('lin_acc_x.csv',ang_vel_x)
csvwrite('lin_acc_y.csv',ang_vel_y)
csvwrite('lin_acc_z.csv',ang_vel_z)
csvwrite('labels.csv',label_impact_noimpact)


