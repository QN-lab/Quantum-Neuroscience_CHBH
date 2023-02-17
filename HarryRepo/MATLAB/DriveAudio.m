clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Time specs
Fs = 22050;
dt = 1/Fs;
stop = 1000;
t = (0:dt:stop-dt);     
%%Sine wave
Fc1 = 1e3; 
Fc2 = 5e3; 

%x = 1.*ones(1,length(t));
 x = 0.001.*sin(2*pi*Fc1*t); %Left Channel
% y = 0.1.*sin(2*pi*Fc2*t); %Right Channel
stereo = [x];%[x;y]; %Two channel stereo
info = audiodevinfo; %Check output needed for amp in ID (Currently ID =3)
ID = 3;
player = audioplayer(stereo,Fs,16,ID);
play(player)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear player