%%Preamble
clear all
close all
clc
ft_defaults
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Acquiring Average Vertical Signal from field without grads

dataset = 'Z:\jenseno-opm\Harry\TL_Grad_Test\20221004\1\20221004_141536_1_1_Z_Square_10mVpp_0.2Hz_raw';
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);

%Ignore y sensor
OPM = [OPM(1,:);OPM(2,:);OPM(3,:);OPM(5,:)];

%Look at signals
sec = length(OPM)./1000;
t = linspace(0,sec,length(OPM)); %Change to seconds
figure(1)
for i = 1:4
    subplot(4,1,i)
    plot(OPM(i,:))
end

%Extracting 3 peaks by eye to calculate the average value (ignoring initial
%spikes as well.

max_ind = [3800:5500, 8500:10750, 13500:15500];
min_ind = [1000:3200, 6000:8200, 11000:13200];

avg_max = mean(OPM(:,max_ind),2);
avg_min = mean(OPM(:,min_ind),2);

Vert_off = avg_max-avg_min; %Vertical Offset (in Tesla)

%Ratio from Volts-> Field
B_ratio_mag = Vert_off./10; %Tesla per milliVolt (~0.6nT/mV)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gradients

clearvars -except B_ratio_mag

filename = '20221004_143950_1_1_Z_Square_10mVpp_0.2Hz_and_dZ_SQ_0.1Vpp_0.05V_off_raw';

dataset = ['Z:\jenseno-opm\Harry\TL_Grad_Test\20221004\1\' filename];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);

%Ignore y sensor
OPM = [OPM(1,:);OPM(2,:);OPM(3,:);OPM(5,:)];

%Look at signals
sec = length(OPM)./1000;
t = linspace(0,sec,length(OPM)); %Change to seconds
figure(2)
for i = 1:4
    subplot(4,1,i)
    plot(OPM(i,:))
end

%Looking at the sensor data, we find the average difference now. 
figure(3)
plot(OPM(4,:))

%Finding indeces by eye
no_grad_ind = [1.25e4:1.4e4, 1.76e4:1.9e4];
yes_grad_ind = [1.42e4:1.49e4, 1.92e4:1.985e4];

avg_no = mean(OPM(:,no_grad_ind),2);
avg_yes = mean(OPM(:,yes_grad_ind),2);

G_field = avg_yes-avg_no; 

%Inputting Sensor location data. 
locs = [-5.6,-0.6,4.4,9.4]; %locations from grid
locs_r = [-18.8,-13.8,-9.0,-3.6]; %measured locations

figure(4)
plot(locs_r,G_field,'r*','MarkerSize',20)
hold on; grid on; axis square
xlabel('Spacial location within shield (cm)')
ylabel('Field at location (T)')
title('100mVpp SQ-wave Gradient wrt Baseline Z field','Fontsize',15)

%Fitting USING ONLY 3 POINTS
fit = polyfit(locs_r(1:3),G_field(1:3)',1);
disp(['slope of linear fit: ', num2str(fit(1)), ' T/cm'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%COMPARING TO RUN WITH DIFFERENT GRADIENT VOLTAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gradients 2
clearvars -except locs locs_r fit 
filename = '20221004_144154_1_1_Z_Square_10mVpp_0.2Hz_and_dZ_SQ_0.05Vpp_0.025V_off_raw';

dataset = ['Z:\jenseno-opm\Harry\TL_Grad_Test\20221004\1\' filename];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);

%Ignore y sensor
OPM = [OPM(1,:);OPM(2,:);OPM(3,:);OPM(5,:)];

%Looking at the sensor data, we find the average difference now. 
figure(5)
plot(OPM(4,:))

%Finding indeces by eye
no_grad_ind2 = [1.08e4:1.2e4, 1.55e4:1.7e4, 2.06e4:2.2e4];
yes_grad_ind2 = [1.22e4:1.29e4, 1.72e4:1.8e4, 2.22e4:2.3e4];

avg_no2 = mean(OPM(:,no_grad_ind2),2);
avg_yes2 = mean(OPM(:,yes_grad_ind2),2);

G_field2 = avg_yes2-avg_no2; 

figure(6)
plot(locs_r,G_field2,'r*','MarkerSize',20)
hold on; grid on; axis square
xlabel('Spacial location within shield (cm)')
ylabel('Field at location (T)')
title('50mVpp SQ-wave Gradient wrt Baseline Z field','Fontsize',15)

%Fitting USING ONLY 3 POINTS
fit2 = polyfit(locs_r(1:3),G_field2(1:3)',1);
disp(['slope of linear fit: ', num2str(fit2(1)), ' T/cm'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gradients 3 (10mV w 5mV off)

clearvars -except locs locs_r fit 
filename = '20221004_144256_1_1_Z_Square_10mVpp_0.2Hz_and_dZ_SQ_0.01Vpp_0.005V_off_raw';

dataset = ['Z:\jenseno-opm\Harry\TL_Grad_Test\20221004\1\' filename];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);

%Ignore y sensor
OPM = [OPM(1,:);OPM(2,:);OPM(3,:);OPM(5,:)];

%Looking at the sensor data, we find the average difference now. 
figure(5)
plot(OPM(4,:))

%Finding indeces by eye
no_grad_ind2 = [1.45e4:1.64e4, 1.95e4:2.14e4, 2.45e4:2.64e4];
yes_grad_ind2 = [1.405e4:1.435e4, 1.905e4:1.935e4, 2.405e4:2.435e4];

avg_no2 = mean(OPM(:,no_grad_ind2),2);
avg_yes2 = mean(OPM(:,yes_grad_ind2),2);

G_field2 = avg_yes2-avg_no2; 

figure(6)
plot(locs_r,G_field2,'r*','MarkerSize',20)
hold on; grid on; axis square
xlabel('Spacial location within shield (cm)')
ylabel('Field at location (T)')
title('50mVpp SQ-wave Gradient wrt Baseline Z field','Fontsize',15)

%Fitting USING ONLY 3 POINTS
fit2 = polyfit(locs_r(1:3),G_field2(1:3)',1);
disp(['slope of linear fit: ', num2str(fit2(1)), ' T/cm'])





