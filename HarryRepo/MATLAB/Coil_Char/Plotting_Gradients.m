%% Generalised Code for gradient analysis
    %May need to go into the code to see if the off period lies within the
    %correct values

clear all
close all
clc
addpath 'Z:\jenseno-opm\fieldtrip-20200331'
% addpath 'Z:\fieldtrip-20200331'
ft_defaults

%% Data files

%File locations
loc1 = 'Z:\jenseno-opm\Data\2022_10_6\1\20221006\1\';

%Filenames
mV_10_5 = '20221006_160530_1_1_10mVpp_0.2Hz__10mVpp_5mVoff_raw';
mV_20_10 = '20221006_160925_1_1_10mVpp_0.2Hz__20mVpp_10mVoff_raw';
mV_50_25 = '20221006_161203_1_1_10mVpp_0.2Hz__50mVpp_25mVoff_raw';
mV_100_50 = '20221006_161450_1_1_10mVpp_0.2Hz__100mVpp_50mVoff_raw';
mV_200_100 = '20221006_161747_1_1_10mVpp_0.2Hz__200mVpp_100mVoff_raw';
mV_250_125 = '20221006_162145_1_1_10mVpp_0.2Hz__250mVpp_125mVoff_raw';

%Acquire Gradients using function (not universal, depends on phase
%difference) 

[mv10grad,SE10_on,SE10_off] = findgradH(mV_10_5,loc1);
[mv20grad,SE20_on,SE20_off] = findgradH(mV_20_10,loc1);
[mv50grad,SE50_on,SE50_off] = findgradH(mV_50_25,loc1);
[mv100grad,SE100_on,SE100_off] = findgradH(mV_100_50,loc1);
[mv200grad,SE200_on,SE200_off] = findgradH(mV_200_100,loc1);
[mv250grad,SE250_on,SE250_off] = findgradH(mV_250_125,loc1);

%Field strength variation (Average of average of on and off for each run)
SE10 = mean([SE10_on,SE10_off]);
SE20 = mean([SE20_on,SE20_off]);
SE50 = mean([SE50_on,SE50_off]);
SE100= mean([SE100_on,SE100_off]);
SE200= mean([SE200_on,SE200_off]);
SE250= mean([SE250_on,SE250_off]);

SEall = [SE10 SE20 SE50 SE100 SE200 SE250];
SE = mean(SEall); 

%Sensor Locations
sloc1 = [-18.8,-14.0,-9.3,-3.8]; loc_err = 0.2.*ones(1,length(sloc1));
SE_ar = SE.*ones(1,length(sloc1));
mat = 0.5.*[mv10grad, mv20grad, mv50grad, mv100grad, mv200grad, mv250grad]';  

%Plot Field Profiles
figure(1); hold on; grid on;
for i = 1:size(mat,1)
    q(i) = plot(sloc1,mat(i,:));
    errorbar(sloc1,mat(i,:),loc_err,'horizontal','k')
    errorbar(sloc1,mat(i,:),SE_ar)
end
xlabel('Position relative to right ledge of platform (cm)')
ylabel('Field Strength (Tesla)')
title('Gradient (dZ/dZ) Profiles at different Currents')


%% Fitting

%Linear Fit
%X Axis
V = [10,20,50,100,200,250]*10^(-3);
index = 1e3.*(V*0.002); %Convert to mAmp

mat = abs(mat)
%Prep Loop
m = zeros(1,size(mat,1)); 
    
for k = 1:size(mat,2)
    temp = polyfit(sloc1,mat(k,:),1);
    m(k) = temp(1);
end
[fit,S] = polyfit(index,m,1);

%Adding data from run w sensors in other direction:
    f_100 = 0.5.*1.6164e-10;
    f_50 = 0.5.*8.0617e-11;
    f_10 = 0.5.*1.6091e-11;

%Input past gradient values for visualisation
    Vi = [10,10,20,50,50,100,100,200,250]*10^(-3);
    index_f = 1e3.*(Vi.*0.002);
    mf = [m(1),f_10,m(2),m(3),f_50,f_100,m(4),m(5),m(6)];

%Doing 3 point fit on the old data
    index_old = [index_f(1),index_f(5),index_f(6)];
    m_old = [f_10,f_50,f_100];
    [fit_old,S_old] = polyfit(index_old,m_old,1);

%Plot Current vs Field Gradient
    A = linspace(index(1),index(length(index)),100);
    [y,delta] = polyval(fit,A,S);
%Old Fit
    A_old = linspace(index_old(1),index_old(length(index_old)),100);
    [y_old,delta_old] = polyval(fit_old,A_old,S_old);

figure(2)
p1 =  plot(index_f,mf,'r*','MarkerSize',10);
    hold on; grid on; axis square;
    title('Field Gradient values for different Currents, with linear fits','FontSize',14)
    ylabel('Gradient (T/cm)','FontSize',14)
    xlabel('P-P Current input into dZ/dZ Gradient Coil (mA)','FontSize',14)
p2 = plot(A,y,'r-');
p3 = plot(A,y+2*delta,'m--');
p4 = plot(A,y-2*delta,'m--');
p5 = plot(A_old,y_old,'b-');
p6 = plot(A_old,y_old+2*delta_old,'m--',A_old,y_old-2*delta_old,'m--');
legend([p1 p2 p5 p3],{'Data',['Fit 1: ' num2str(fit(1)) 'T/cm*mA']...
    ,['Fit 2: ' num2str(fit_old(1)) 'T/cm*mA'],'95% Confidence'})


