%%Preamble
clear all
close all
clc
%addpath('/Users/Harry/Desktop/All/Archive/Summer_Studentship/fieldtrip/')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% No Applied Fields
%Read in FT
dataset = '/Volumes/jenseno-opm/Harry/TL_1/20220930_163539_TL_TL_noise_3minNoise_raw';
%MAC^
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Look at OPMs
figure(1)
subplot(2,1,1) %Sampling Rate to get time??
plot(OPM(1,:)) %Looks like maybe the lift was running?
    xlabel('Time increment')
    ylabel('B Field')
subplot(2,1,2)
plot(OPM(2,:))
    xlabel('Time increment')
    ylabel('B Field')
    
%Spectrum
fs=1000;
pxx=[];
f=[];

figure(2);
clf;
box on
hold on
for ch=1:2
    [pxx,f] = pwelch(OPM(ch,:),hann(2000),1000,2000,fs);
    plot(f,sqrt(pxx),'LineWidth',2)
end
hold off
grid on
xlim([0 100])
ylim([0 1.1e-11])
xlabel('Hz')
ylabel('Noise floor T/sqrt(Hz)')
legend('Ch01','Ch02','Ch03','Ch04','Ch05')
set(gca, 'YScale', 'log')
set(gca,'fontsize',12,'FontWeight','bold','LineWidth',1.5,'TickDir','both')

%% Looking at Elevator Run

dataset = 'Z:\jenseno-opm\Harry\TL_1\20220929_181716_Z_TL_empty_shield_elevators_raw';

cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Look at OPMs
figure(3)
subplot(2,1,1) 
plot(OPM(2,:))
    xlabel('Time increment')
    ylabel('B Field')
subplot(2,1,2)
plot(OPM(3,:))
    xlabel('Time increment')
    ylabel('B Field')

%% Z_3 Run (increasing frequency, file somewhere on the Lab PC)

dataset = 'Z:\jenseno-opm\Harry\TL_1\20220929_180738_Z_TL_Z_3_raw';
%Read in FT
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Look at OPMs
figure(4)
subplot(2,1,1)
plot(OPM(2,:))
    xlabel('Time increment')
    ylabel('B Field')
subplot(2,1,2)
plot(OPM(3,:))
    xlabel('Time increment')
    ylabel('B Field')

mA_2 = OPM(2:3,1.4e4:2.2e4);
mA_5 = OPM(2:3,2.6e4:3.4e4);
mA_10 = OPM(2:3,3.9e4:4.7e4);
mA_25 = OPM(2:3,5.2e4:6.2e4);

m2 = max(mA_2,[],2);
m5 = max(mA_5,[],2);
m10 = max(mA_10,[],2);
m25 = max(mA_25,[],2);

maxVals = [m2,m5,m10,m25];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




