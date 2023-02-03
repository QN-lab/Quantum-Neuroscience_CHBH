
%Preamble
clear all
close all
clc
%% Values
%Measuring the voltage/current ratios of the TL coils. 
%Zurich

Vz = [0.918,-0.918,1.92,-1.92,4.93,-4.93,9.94,-9.94,25,-24.96,49.9,-49.9,74.98,-74.98,99.9,-99.9,250,-250,500,-500,1000,-1000]*10^(-3);
Az = [2.1,-1.0,4.1,-3.0,10.1,-9.0,20.0,-19.0,49.9,-48.8,99.4,-98.6,149.2,-148.6,198.8,-198.2,497.6,-497.9,994.6,-996.8,1989.3,-1994.3]*10^(-6);

%Sig Generator

%V is pp Voltage
Vs = [2,-2,-5,5,10,-10,-25,25,50,-50,-100,100,500,-500,-1000,1000]*10^(-3);
As = [4.1,-3.3,-9.2,10.0,19.9,-19.2,-49.0,49.6,99.1,-98.7,-198.3,198.3,994.0,-996.4,-1990.3,1987.3]*10^(-6);


%Plotting

figure(1)
subplot(1,2,1)
plot(Vz,Az,'b*'); hold on
    xlabel('Voltage','Fontsize',17)
    ylabel('Current','Fontsize',17)
    title('Zurich V vs A')
    grid on; axis square;
    
    
Pz = polyfit(Vz,Az,1);
Pzfit = polyval(Pz,Vz);
plot(Vz,Pzfit,'r-')

subplot(1,2,2)
plot(Vs,As,'b*'); hold on
    xlabel('Voltage','Fontsize',17)
    ylabel('Current','Fontsize',17)
    title('Signal Generator V vs A')
    grid on; axis square;
    
    
Ps = polyfit(Vs,As,1);
Psfit = polyval(Ps,Vs);
plot(Vs,Psfit,'r-')



