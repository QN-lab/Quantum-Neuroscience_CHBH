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
%loc = 'Z:\jenseno-opm\Data\2022_10_24\1\20221024\2\';
loc = 'Z:\jenseno-opm\Data\2022_10_26\1\20221026\1\';
filename    = '20221026_125141_1_1_Bz=0.5nT-8Hz_dBz=2pT-13Hz_raw'; 
dataset     = [loc filename];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Signal
Fs = 1000;
grad = OPM(2,:)-OPM(1,:);

figure(1)
subplot(2,1,1)
plot(1:length(OPM),OPM);hold on;
    title('Raw OPM Signal')

subplot(2,1,2)
plot(1:length(grad),grad); hold on;
    title('Raw Syn. Grad. Signal')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Power
figure(2);
clf;
box on
hold on
sqrpo = [];
for ch=1:2
    [pxxo,fo] = pwelch(OPM(ch,:),hann(2000),1000,2000,Fs);
    plot(fo,sqrt(pxxo),'LineWidth',2)
    sqrpo(ch,:) = sqrt(pxxo);
end
[pxxg,fg] = pwelch(grad,hann(2000),1000,2000,Fs);
plot(fg,sqrt(pxxg)./4,'g','LineWidth',2)
    hold on
    grid on
    xlim([0 100])
    ylim([0 1.1e-8])
    xlabel('Hz')
    legend('Ch01','Ch02','Grad')
    ylabel('Noise floor T/sqrt(Hz)')
    set(gca, 'YScale', 'log')
    set(gca,'fontsize',12,'FontWeight','bold','LineWidth',1.5,'TickDir','both')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Analysis
asd_o = mean(sqrpo,1);
asd_g = (sqrt(pxxg)./4)';

Bz_peak = 8;
dBz_peak = 13;

    %Find Value of the Peaks (8,13 Hz)
    indx_Bz = find(fo == Bz_peak); %fo = fg
    indx_dBz = find(fo == dBz_peak);
    
    %Find these peaks in both sensor types
    Bz_peak = [asd_o(indx_Bz), asd_g(indx_Bz)];
    dBz_peak = [asd_o(indx_dBz), asd_g(indx_dBz)];
    
    %Ratio of Peak Ratios
    Ratio = (Bz_peak(1)./dBz_peak(1))./(Bz_peak(2)./dBz_peak(2));
    
%Noise 
    %mean of 3-6 and 90-100Hz, rest of the floor will lie between these
    %vals

    in3_6 = find(fo<=6 & fo>=3);
    in60_70 = find(fo<=70 & fo>=60);
    in58_62 = find(fo<=62 & fo>=58);
    in66_70 = find(fo<=70 & fo>=66);
    in90_94 = find(fo<=94 & fo>=90);
    in98_100 = find(fo<=100 & fo>=98);
    in90_100 = find(fo<=100 & fo>=90);
    
%     n_ind = [in3_6;in60_70;in90_100]';
    n_ind = [in3_6;in58_62;in66_70;in90_94;in98_100]';
    
    avg_Noise = [mean(asd_o(n_ind)), mean(asd_g(n_ind))];
    
    SNRo = dBz_peak(1)./avg_Noise(1);
    SNRg = dBz_peak(2)./avg_Noise(2);

%Ratio b/w Peaks

    
%% Noise vs Signal Analysis
clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath 'Z:\fieldtrip-20200331';
addpath 'Z:\jenseno-opm\fieldtrip-20200331';
ft_defaults

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%loc = 'Z:\Data\2022_10_19\1\20221019\1\';
%loc = 'Z:\jenseno-opm\Data\2022_10_24\1\20221024\2\';
loc = 'Z:\jenseno-opm\Data\2022_10_26\1\20221026\1\';
filename    = '20221024_124756_2_1_both_dBz=3.07mV_raw'; 
dataset     = [loc filename];
cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM         = ft_read_data(cfg.dataset);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 1000;
grad = OPM(2,:)-OPM(1,:);

figure(1)
subplot(2,1,1)
    plot(1:length(OPM),OPM);hold on;
    title('Raw OPM Signal')    
subplot(2,1,2)
    plot(1:length(grad),grad); hold on;
    title('Raw Syn. Grad. Signal')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
clf;
box on
hold on
sqrpo = [];
for ch=1:2
    [pxxo,fo] = pwelch(OPM(ch,:),hann(2000),1000,2000,Fs);
    plot(fo,sqrt(pxxo),'LineWidth',2)
    sqrpo(ch,:) = sqrt(pxxo);
end
[pxxg,fg] = pwelch(grad,hann(2000),1000,2000,Fs);
plot(fg,sqrt(pxxg)./4,'g','LineWidth',2)
sqrpg = sqrt(pxxg)';
    hold off
    grid on
    xlim([0 100])
    ylim([0 1.1e-8])
    xlabel('Hz')
    legend('Ch01','Ch02','Grad')
    ylabel('Noise floor T/sqrt(Hz)')
    set(gca, 'YScale', 'log')
    set(gca,'fontsize',12,'FontWeight','bold','LineWidth',1.5,'TickDir','both')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






