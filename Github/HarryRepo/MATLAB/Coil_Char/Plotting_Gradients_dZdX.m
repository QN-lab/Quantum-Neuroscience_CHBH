%% Generalised Code for gradient analysis
    %May need to go into the code to see if the off period lies within the
    %correct values

clear all
close all
clc
addpath 'Z:\jenseno-opm\fieldtrip-20200331'
%addpath 'Z:\fieldtrip-20200331'
ft_defaults

%% Data files

%File locations
loc1 = 'Z:\jenseno-opm\Data\2022_10_12\1\20221012\1\';

%Filenames
mV_10_5 = '20221012_111352_1_1_10mVpp_0.2Hz_10mVpp_5off_raw';
mV_20_10 = '20221012_111557_1_1_10mVpp_0.2Hz_20mVpp_10off_raw';
mV_50_25 = '20221012_111713_1_1_10mVpp_0.2Hz_50mVpp_25off_raw';
mV_100_50 = '20221012_111834_1_1_10mVpp_0.2Hz_100mVpp_50off_raw';
mV_200_100 = '20221012_112000_1_1_10mVpp_0.2Hz_200mVpp_100off_raw';
mV_250_125 = '20221012_112131_1_1_10mVpp_0.2Hz_250mVpp_125off_raw';

%Acquire Gradients using function (not universal, depends on phase
%difference) 

[mv10grad,SE10_on,SE10_off] = findgradH_dZdX(mV_10_5,loc1);
[mv20grad,SE20_on,SE20_off] = findgradH_dZdX(mV_20_10,loc1);
[mv50grad,SE50_on,SE50_off] = findgradH_dZdX(mV_50_25,loc1);
[mv100grad,SE100_on,SE100_off] = findgradH_dZdX(mV_100_50,loc1);
[mv200grad,SE200_on,SE200_off] = findgradH_dZdX(mV_200_100,loc1);
[mv250grad,SE250_on,SE250_off] = findgradH_dZdX(mV_250_125,loc1);

%Field strength variation (Average of average of on and off for each run)
SE10 = mean([SE10_on,SE10_off]);
SE20 = mean([SE20_on,SE20_off]);
SE50 = mean([SE50_on,SE50_off]);
SE100= mean([SE100_on,SE100_off]);
SE200= mean([SE200_on,SE200_off]);
SE250= mean([SE250_on,SE250_off]);

SEall = [SE10 SE20 SE50 SE100 SE200 SE250];
SE = mean(SEall); 

%Sensor Locations (first sensor is at highest location):
    %adding 13mm for sensitive location within sensor
sloc1 = [-4.2,-2.1,0,2.1,4.2]; loc_err = 0.1.*ones(1,length(sloc1));
SE_ar = SE.*ones(1,length(sloc1));
%IMPORTANT: DIVIDE BY 2 BECAUSE PP VOLTAGE CORRESPONDS TO HALF OF THE LIVE
%CURRENT
mat = 0.5.*[mv10grad, mv20grad, mv50grad, mv100grad, mv200grad, mv250grad]'; 

%Plot Field Profiles
figure(1); hold on; grid on;
for i = 1:5
    q(i) = plot(sloc1,mat(i,:));
    errorbar(sloc1,mat(i,:),loc_err,'horizontal','k')
    errorbar(sloc1,mat(i,:),SE_ar)
end
xlabel('Horizontal Position(cm)')
ylabel('Field Strength (Tesla)')
title('Gradient (dZ/dX) Profiles at different Currents')

%% Fitting

%Linear Fit
%X Axis
V = [10,20,50,100,200,250]*10^(-3);
index = 1e3.*(V*0.002); %Convert to mAmp (factor of 0.5 is to go from pp Voltage to Current)

%Prep Loop
m = zeros(1,size(mat,1)); 
    
for k = 1:size(mat,1)
    temp = polyfit(sloc1,mat(k,:),1);
    m(k) = temp(1);
end
%m = abs(m); %Abs for fitting w old data 

    [fit,S] = polyfit(index,m,1);

%Plot Current vs Field Gradient
    A = linspace(index(1),index(length(index)),100);
    [y,delta] = polyval(fit,A,S);

figure(2)
p1 = plot(index,m,'b*','MarkerSize',10);
hold on; grid on; axis square;
    title('Field Gradient values for different Currents, with linear fits','FontSize',14)
    ylabel('Gradient (T/cm)','FontSize',14)
    xlabel('P-P Current input into dZ/dX Gradient Coil (mA)','FontSize',14)
p2 = plot(A,y,'r-');
p3 = plot(A,y+2*delta,'m--');
p4 = plot(A,y-2*delta,'m--');
legend([p1 p2 p3 ],{'Data'...
    ['Fit: ' num2str(fit(1)) 'T/(cm*mA)'],'95% Confidence'},'Location','northeast')


