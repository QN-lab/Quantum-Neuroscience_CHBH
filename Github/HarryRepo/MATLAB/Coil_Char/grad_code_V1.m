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

filename = '20221010_173701_1_1_10mVpp_0.2Hz_50mVpp_25off_re_raw';
dataset = ['Z:\jenseno-opm\Data\2022_10_10\1\20221010\1\' filename];
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


%Finding where trigger changes 

delta = diff(hbool);

pre_trig = delta == 1; %Spikes to 1 when signal is about to be measured
indx = find(pre_trig == 1); 

mat_on = zeros(length(indx),1181); %Picked values past trigger to include
mat_off = zeros(length(indx),1111);
for i = 1:length(indx)
    ino = indx(i)+20:indx(i)+1200;
    mat_on(i,:) = ino;
    inf = indx(i)+1320:indx(i)+2430;
    mat_off(i,:) = inf;
end

on_in = reshape(mat_on,1,[]);
bound = on_in < length(OPM); %Remove values exceeding bounds
on = on_in(bound);

off_in = reshape(mat_off,1,[]);
bound2 = off_in < length(OPM); %Remove values exceeding bounds
off = off_in(bound2);

grad_on = OPM(:,on);
grad_off = OPM(:,off);

mean_on = mean(grad_on,2);
mean_off = mean(grad_off,2);

grad = mean_on-mean_off;


%% Errors

stderror_on = std(grad_on(:))./sqrt(numel(grad_on));
stderror_off = std(grad_off(:))./sqrt(numel(grad_off));


