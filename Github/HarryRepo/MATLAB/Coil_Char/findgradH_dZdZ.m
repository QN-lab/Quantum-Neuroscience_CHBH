function [grad,SE_on,SE_off] = findgradH_dZdY(filename,loc)

dataset = [loc filename];
addpath(dataset)

cfg         = [];
cfg.dataset = [dataset '.fif'];
hdr         = ft_read_header(cfg.dataset);
OPM        = ft_read_data(cfg.dataset);

%Ignore trigger for these runs
 %OPM 2 needs to be in the 4th position

delta = diff(OPM(2,:));
delta(end+1) = 0;

indx = find(delta>1.6e-9 &delta<2e-9);

mat_on = zeros(length(indx),1031); %Picked values past trigger to include
mat_off = zeros(length(indx),1071);

%Applying indeces
for i = 1:length(indx)
    ino = indx(i)+50:indx(i)+1080;
    mat_on(i,:) = ino;
    inf = indx(i)+1330:indx(i)+2400;
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

