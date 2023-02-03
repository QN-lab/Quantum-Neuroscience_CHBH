clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FieldTrip
%addpath 'Z:\fieldtrip-20200331';
addpath 'Z:\jenseno-opm\fieldtrip-20200331';
ft_defaults
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load Data
%loc = 'Z:\Data\2022_10_19\1\20221019\1\';
loc = 'Z:\jenseno-opm\Data\2022_10_28\1\20221028\1\';
filename    = '20221028_112427_1_1_5chNoise_Zdir_raw'; 
dataset     = [loc filename];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Signal
Fs = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Power
figure(2);
clf;
box on
hold on
sqrpo = [];
for ch=1:size(OPM,1)
    [pxxo,fo] = pwelch(OPM(ch,:),hann(2000),1000,2000,Fs);
    plot(fo,sqrt(pxxo),'LineWidth',2)
    sqrpo(ch,:) = sqrt(pxxo);
end
    hold off
    grid on
    xlim([0 100])
    ylim([0 1.1e-8])
    xlabel('Hz')
    legend('Ch01','Ch02','Ch03','Ch04','Ch05')
    ylabel('Noise floor T/sqrt(Hz)')
    set(gca, 'YScale', 'log')
    set(gca,'fontsize',12,'FontWeight','bold','LineWidth',1.5,'TickDir','both')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





