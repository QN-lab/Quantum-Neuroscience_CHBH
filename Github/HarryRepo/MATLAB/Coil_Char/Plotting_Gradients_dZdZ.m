%% Generalised Code for gradient analysis
    %May need to go into the code to see if the off period lies within the
    %correct values

clear all
close all
clc
%addpath 'Z:\jenseno-opm\fieldtrip-20200331'
addpath 'Z:\fieldtrip-20200331'
ft_defaults

%% Data files

%File locations
loc1 = 'Z:\Data\2022_10_17\1\20221017\1\';

%Filenames
L5 = '20221017_160712_1_1_10mV_0.2Hz_5mV_2-5off_raw';
L10 = '20221017_160543_1_1_10mV_0.2Hz_10mV_5off_raw';
L20 = '20221017_160429_1_1_10mV_0.2Hz_20mV_10off_raw';
L50 = '20221017_160247_1_1_10mV_0.2Hz_50mV_25off_raw';
L100 = '20221017_160134_1_1_10mV_0.2Hz_100mV_50off_raw';
L200 = '20221017_155921_1_1_10mV_0.2Hz_200mV_100off_raw';
L250 = '20221017_155722_1_1_10mV_0.2Hz_250mV_125off_raw';

%Acquire Gradients using function (not universal, depends on phase
%difference) 
[L5grad,SE5_on,SE5_off] = findgradH_dZdZ(L5,loc1);
[L10grad,SE10_on,SE10_off] = findgradH_dZdZ(L10,loc1);
[L20grad,SE20_on,SE20_off] = findgradH_dZdZ(L20,loc1);
[L50grad,SE50_on,SE50_off] = findgradH_dZdZ(L50,loc1);
[L100grad,SE100_on,SE100_off] = findgradH_dZdZ(L100,loc1);
[L200grad,SE200_on,SE200_off] = findgradH_dZdZ(L200,loc1);
[L250grad,SE250_on,SE250_off] = findgradH_dZdZ(L250,loc1);

%Field strength variation (Average of average of on and off for each run)
SE5 = mean([SE5_on,SE5_off]);
SE10 = mean([SE10_on,SE10_off]);
SE20 = mean([SE20_on,SE20_off]);
SE50 = mean([SE50_on,SE50_off]);
SE100= mean([SE100_on,SE100_off]);
SE200= mean([SE200_on,SE200_off]);
SE250= mean([SE250_on,SE250_off]);

%File locations
loc2 = 'Z:\Data\2022_10_17\1\20221017\2\';

%Filenames
R5 = '20221017_164617_2_1_Rev_10mV_0.2Hz_5mV_2-5off_raw';
R10 = '20221017_164507_2_1_Rev_10mV_0.2Hz_10mV_5off_raw';
R20 = '20221017_164404_2_1_Rev_10mV_0.2Hz_20mV_10off_raw';
R50 = '20221017_164243_2_1_Rev_10mV_0.2Hz_50mV_25off_raw';
R100 = '20221017_164051_2_1_Rev_10mV_0.2Hz_100mV_50off_raw';
R200 = '20221017_163930_2_1_Rev_10mV_0.2Hz_200mV_100off_raw';
R250 = '20221017_163646_2_1_Rev_10mV_0.2Hz_250mV_125off_raw';

%Acquire Gradients using function (not universal, depends on phase
%difference) 

[R5grad,RSE5_on,RSE5_off] = findgradH_dZdZ(R5,loc2);
[R10grad,RSE10_on,RSE10_off] = findgradH_dZdZ(R10,loc2);
[R20grad,RSE20_on,RSE20_off] = findgradH_dZdZ(R20,loc2);
[R50grad,RSE50_on,RSE50_off] = findgradH_dZdZ(R50,loc2);
[R100grad,RSE100_on,RSE100_off] = findgradH_dZdZ(R100,loc2);
[R200grad,RSE200_on,RSE200_off] = findgradH_dZdZ(R200,loc2);
[R250grad,RSE250_on,RSE250_off] = findgradH_dZdZ(R250,loc2);

RSE5 = mean([RSE5_on,RSE5_off]);
RSE10 = mean([RSE10_on,RSE10_off]);
RSE20 = mean([RSE20_on,RSE20_off]);
RSE50 = mean([RSE50_on,RSE50_off]);
RSE100= mean([RSE100_on,RSE100_off]);
RSE200= mean([RSE200_on,RSE200_off]);
RSE250= mean([RSE250_on,RSE250_off]);

SEall = [SE5 SE10 SE20 SE50 SE100 SE200 SE250 RSE5 RSE10 RSE20 RSE50 RSE100 RSE200 RSE250];

SE = mean(SEall);

%Sensor Locations (first sensor is at highest location):
    %adding 13mm for sensitive location within sensor
sloc1 = [-3.4,-0.4,2.6]; loc_err = 0.1.*ones(1,length(sloc1));
sloc2 = [-2.6,0.4,3.4];
SE_ar = SE.*ones(1,length(sloc1));

mat1 = 0.5.*[L5grad, L10grad, L20grad, L50grad, L100grad, L200grad, L250grad]'; 
mat2 = 0.5.*[R5grad, R10grad, R20grad, R50grad, R100grad, R200grad, R250grad]';

%Plot Field Profiles
figure(1); hold on; grid on;
for i = 1:size(mat1,1)
    q(i) = plot(sloc1,mat1(i,:));
    plot(sloc2,mat2(i,:))
%     errorbar(sloc1,mat1(i,:),loc_err,'horizontal','k')
%     errorbar(sloc1,mat1(i,:),SE_ar)
%     errorbar(sloc2,mat2(i,:),loc_err,'horizontal','k')
%     errorbar(sloc1,mat2(i,:),SE_ar)
end
xlabel('Z Position(cm)')
ylabel('Field Strength (Tesla)')
title('Gradient (dZ/dZ) Profiles at different Currents')

%% Fitting

%Linear Fit
%X Axis
V = [5,10,20,50,100,200,250]*10^(-3);
index = 1e3.*(V*0.002); %Convert to mAmp

sloc = [sloc1 sloc2];
mat = [mat1 mat2];
%Prep Loop
m = zeros(1,size(mat,1)); 
    
for k = 1:size(mat,1)
    temp = polyfit(sloc,mat(k,:),1);
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
    xlabel('P-P Current input into dZ/dZ Gradient Coil (mA)','FontSize',14)
p2 = plot(A,y,'r-');
p3 = plot(A,y+2*delta,'m--');
p4 = plot(A,y-2*delta,'m--');
legend([p1 p2 p3 ],{'Data'...
    ['Fit: ' num2str(fit(1)) 'T/(cm*mA)'],'95% Confidence'},'Location','northeast')


