clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delete(instrfind({'Port'},{'COM9'}));

s = serialport('COM9',115200);

t=0:.001:1;
f=50;
sq=0.3*0.5*(square(2*pi*f*t)+1);

tic
for i = 1:length(t)
resp = writeread(s,['!set;5;4mA;', num2str(sq(i))]);
end
toc