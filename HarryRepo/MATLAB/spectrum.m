%%%YULIA'S CODE:

clear all
close all
clc
addpath '/Volumes/jenseno-opm/fieldtrip-20200331'
ft_defaults

%loc = 'Z:\Data\2022_10_21_noise_for_MSL\20221021\1\';
loc = '/Volumes/jenseno-opm/Data/2022_10_24/1/20221024/2/';
fname = '20221024_124756_2_1_both_dBz=3.07mV_raw';
dataset = [loc fname];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
data_spect_subject        = ft_read_data(cfg.dataset);
%%
fs=1000;
pxx=[];
f=[];

figure(1);
clf;
box on
hold on
for ch=1:2
    [pxx,f] = pwelch(data_spect_subject(ch,:),hann(1000),1000,2000,fs);
    plot(f,sqrt(pxx),'LineWidth',2)
end
hold off
grid on
xlim([0 100])
ylim([0 1.1e-8])
xlabel('Hz')
ylabel('Noise floor T/sqrt(Hz)')
legend('Ch01','Ch02','Ch03','Ch04','Ch05')
set(gca, 'YScale', 'log')
set(gca,'fontsize',12,'FontWeight','bold','LineWidth',1.5,'TickDir','both')
