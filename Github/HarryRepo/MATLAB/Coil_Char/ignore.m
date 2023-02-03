%%Preamble
clear all
close all
clc
addpath 'Z:\jenseno-opm\fieldtrip-20200331'
%addpath 'Z:\fieldtrip-20200331'
ft_defaults

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ANALYSING RUNS FROM SENSORS IN OPPOSITE DIRECTIONS
%% Acquiring Average Vertical Signal from field without grads

filename = '20221006_160530_1_1_10mVpp_0.2Hz__10mVpp_5mVoff_raw';
dataset = ['Z:\jenseno-opm\Data\2022_10_6\1\20221006\1\' filename];
%dataset = ['Z:\Data\2022_10_6\1\20221006\1\' filename];
addpath(dataset)

cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPMi        = ft_read_data(cfg.dataset);

%Ignore y sensor
trig = OPMi(1,:);
OPM = OPMi([2:4,6],:);

%10mV wave is always the same (be
grad_off_bool = (OPM>= 5.95e-9 & OPM<= 6.05e-9);
grad_off = OPM.*grad_off_bool;

figure(1)

for i = 1:size(OPM,1)
    subplot(size(OPM,1),1,i)
    plot(OPM(i,:))
    hold on
end

figure(2)
hbool = trig>mean(trig); %hbool occurs during the time of interest
subplot(3,1,1)
plot(hbool); ylim([-0.2,1.2]); %hbool is basically a 

roi = hbool.*trig; %check there are no intermediary values
subplot(3,1,2)
plot(roi,'b.')
subplot(3,1,3)
plot(OPM(1,:))

%'Active Area'
act_bool = hbool.*ones(size(OPM));
act_sig = act_bool.*OPM;

%Find non-zero data in active region (gradient on)
nzbool_on = (act_sig ~= grad_off & act_sig > 0);
grad_on = nzbool_on.*OPM;

%Find abrupt changes in the active region
deriv = diff(act_sig,[],2);
deriv(:,end+1) = 0;

figure(3)
for i = 1:size(OPM,1)
    subplot(size(OPM,1),1,i)
    plot(act_sig(i,:))
    hold on
    plot(grad_off(i,:),'r')
    plot(grad_on(i,:),'b')
end

%Average strength of no gradient
nzbool_off = (grad_off > 0);
for i = 1:size(grad_off,1)
    avg_off(i,:) = mean(grad_off(i,nzbool_off(i,:)));
end


for i = 1:size(grad_off,1)
    avg_on(i,:) = mean(grad_on(i,nzbool_on(i,:)));
end

%% Plotting

sloc = [-18.8,-14.0,-9.3,-3.8];











