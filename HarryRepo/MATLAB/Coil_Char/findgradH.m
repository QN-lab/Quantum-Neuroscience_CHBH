function [grad,SE_on,SE_off] = findgradH(filename,loc)

dataset = [loc filename];
addpath(dataset)

cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPMi        = ft_read_data(cfg.dataset);

%Ignore y sensor
trig = OPMi(1,:);
OPM = OPMi([2:4,6],:);


hbool = trig>mean(trig); %hbool occurs during the time of interest

delta = diff(hbool); %Finding where trigger changes 

pre_trig = delta == 1; %Spikes to 1 when signal is about to be measured
indx = find(pre_trig == 1); 

mat_on = zeros(length(indx),1181); %Picked values past trigger to include
mat_off = zeros(length(indx),1111);
for i = 1:length(indx)
    ino = indx(i)+20:indx(i)+1200;
    mat_on(i,:) = ino;
    inf = indx(i)+1320:indx(i)+2430;
    mat_off(i,:) = inf;
end

on_in = reshape(mat_on,1,[]);
bound = on_in < length(OPM); %Remove values exceeding bounds
on = on_in(bound);

off_in = reshape(mat_off,1,[]);
bound2 = off_in < length(OPM); %Remove values exceeding bounds
off = off_in(bound2);

grad_on = OPM(:,on);
grad_off = OPM(:,off);

mean_on = mean(grad_on,2);
mean_off = mean(grad_off,2);

SE_on = std(grad_on(:))./sqrt(numel(grad_on));
SE_off = std(grad_off(:))./sqrt(numel(grad_off));

grad = mean_on-mean_off;
end 

