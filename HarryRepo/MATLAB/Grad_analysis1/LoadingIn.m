clear all 
close all
clc

%% Load in data
addpath('Z:\Data\2022_11_18')
FileData = load('First_go_grad_closed_00000.mat');
 
for i = 1:4
    header(i,:) = FileData.dev3994.demods.sample{1,i}.header.name;
    xdata(i,:) = FileData.dev3994.demods.sample{1,i}.x;
    ydata(i,:) = FileData.dev3994.demods.sample{1,i}.y;
    freq(i,:) = FileData.dev3994.demods.sample{1,i}.frequency;
    phase(i,:) = FileData.dev3994.demods.sample{1,i}.phase;
end

%% Plotting
%Plot Data
figure(1)
title('X and Y Data')
for i = 1:4
    subplot(4,1,i)
    plot(freq(i,:),xdata(i,:));hold on
    plot(freq(i,:),ydata(i,:))
end

% figure(2)
% title('frequency')
% for i = 1:4
%     subplot(4,1,i)
%     plot(freq(i,:));hold on
% end

figure(3)
title('Phase')
for i = 1:4
    subplot(4,1,i)
    plot(phase(i,:));hold on
end

%Fitting using lorentzfit.m file from exchange:

[y1, PARAMS1]= ...
            lorentzfit(1:length(xdata),(xdata(1,:)),[0,0,13,0]);
                    
[y2, PARAMS2]= ...
                        lorentzfit(1:length(xdata),(xdata(4,:)));                   


figure(4)
for i = 1:4
    subplot(4,1,i)
    plot(freq(i,:),xdata(i,:)); hold on
    L_fit(i,:) = lorentzfit(freq(i,:),xdata(i,:));
    plot(freq(i,:),L_fit(i,:))
end

%%First two fit well, second 2 dont
% Find the half max value.
halfMax = (min(L_fit(1,:)) + max(L_fit(1,:))) / 2;
% Find where the data first drops below half the max.
index1 = find(L_fit(1,:) >= halfMax, 1, 'first');
% Find where the data last rises above half the max.
index2 = find(L_fit(1,:) >= halfMax, 1, 'last');
fwhm = index2-index1 + 1; % FWHM in indexes.
% OR, if you have an x vector
fwhmx = freq(index2) - freq(index1);

HM2 = max(L_fit(2,:))./2;




