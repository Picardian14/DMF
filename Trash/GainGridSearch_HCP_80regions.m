clear all;
close all;
basefold = './data/';
data_file = 'hcp1003_REST1_LR_dbs80'; % 
load('data/heterogeneitys/myelin_aal.mat') % Heterogenity
load('data/heterogeneitys/SC_dbs80HARDIFULL.mat') % Heterogenity
load([basefold,data_file,'.mat'])

SC = SC_dbs80HARDI;

%% 

C = SC/max(max(SC))*0.2;
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.04; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = 0.07; % high cut-off of the bandpass filter 0.1
params.wsize = 30; % size of the FCD windows
params.overlap = 28; % overlap of the FCD windows
% IT IS TR=2 FOR ENZOS SLEEP DATA
params.TR = 2; % repetition time of the fMRI signal (will be used to simulate fMRI)
params.batch_size = 50000; % batch for
% circular buffer of simulation
% params.seed = 10; % initial condition for the simulation. 
% Heterogenity
params.receptors = av/max(av);
% Setting data constants
params.N=80;
params.NSUB=398;
params.TMAX=464;
params.START_TIME=101;
Isubdiag = find(tril(ones(params.N),-1));

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=subject{nsub}.dbs80ts(:, params.START_TIME:params.START_TIME+params.TMAX) ; % TS_W tiene registros de distinta longitud
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto

end

WFCdata = permute(WFCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
ave_fc = mean(WFCdataF,3);

% Optimizable parameters
prev_res = load('Results/G_HCP/SSIM/SSIM.mat');
params.G = prev_res.best_pars_ssim.G; 
metric = 'ssim'
wgain = [1.2 1.6]; % Here only optimizing gain
bgain = [-0.2 -0.1];
% Number of points in grid = n_weigth*n_bias
n_weigth = 40;
n_bias = 10;


checkpoint_folder = 'checkpoints/';
experiment_name = 'GainGridSearch_HCP';
if ~exist(fullfile("Figuras",experiment_name))
    mkdir(fullfile("Figuras",experiment_name))
end
if ~exist(fullfile("Results",experiment_name))
    mkdir(fullfile("Results",experiment_name))
end
   
if ~exist(fullfile("data/checkpoints",experiment_name))
    mkdir(fullfile("data/checkpoints",experiment_name))
end
opt_time_1 = 1800; % 30 min
opt_time_2 = 900; % 15 min
%opt_time_1 = 420; % 2 min
%opt_time_2 = 60; % 1 min

% Setting DMF parameters
N = size(params.C,1);
stren = sum(params.C);
isubfc = find(tril(ones(N),-1));
nsteps = params.TMAX.*(1000); % number of DMF timepoints
gamma_ent_fun = @(a) a(1) + log(a(2)) + log(gamma(a(1))) + (1-a(1))*psi(a(1));


opt_vars = [];


if length(wgain(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    wgainvals = optimizableVariable('wgain',[wgain(1) wgain(2)]);
    opt_vars = [wgainvals];
end

if length(bgain(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    bgainvals = optimizableVariable('bgain',[bgain(1) bgain(2)]);
    opt_vars = [opt_vars bgainvals];
end


weight_space = linspace(wgain(1), wgain(2), n_weigth);
bias_space = linspace(bgain(1), bgain(2), n_bias);
weight_range = 1:length(weight_space);
bias_range = 1:length(bias_space);
local_fc_error = ones(length(weight_space), length(bias_space));
local_fc = zeros(length(weight_space), length(bias_space), N, N);



parfor cur_weight=weight_range
    thispars=params;       
    %local_min = 1;    
    for cur_bias=bias_range        
        
        thispars.wgain = weight_space(cur_weight);                                           
        thispars.bgain = bias_space(cur_bias);                 
        % Simulating        
        [rates,bold] = DMF_hetero(thispars, nsteps,'both'); % runs simulation
        bold = bold(:,params.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        if isempty(bold)            
            local_fc_error(cur_weight, cur_bias) = nan;                    
        end
        % Filtering and computing FC
        filt_bold = filter_bold(bold',params.flp,params.fhi,params.TR);
        sim_fc = corrcoef(filt_bold);
        
        % Computing rates
        rates = rates(:,(params.burnout*1000):end);
        reg_fr = mean(rates,2);
        reg_ent = zeros(N,1);
        for n=1:N
            gamma_pars = gamfit(rates(n,:));
            reg_ent(n) = gamma_ent_fun(gamma_pars);
        end
    
        % Computing FC error: Mean Squared differences between vectorized FCs              
        mse_error= mean((sim_fc(isubfc)-ave_fc(isubfc)).^2); % MSE FC
        ssim_error = 1-ssim(ave_fc,sim_fc);
        corr_error = 1-corr2(ave_fc, sim_fc);        
        metrics=containers.Map;        
        metrics('ssim')=ssim_error;        
        
        %parsave(cur_weight,cur_bias, sim_fc, metrics(metric))
    
        local_fc_error(cur_weight, cur_bias)=metrics(metric);        
        local_fc(cur_weight, cur_bias, :, :) = sim_fc;
    
        
    end
end
disp("Finished 1st run ok")
[fc_error_bias, weight] = min(local_fc_error);
[fc_error, bias] = min(fc_error_bias);
weight_out = weight_space(weight(bias));                                           
bias_out = bias_space(bias); 
outdata = {squeeze(local_fc(weight(bias), bias, :, :)), weight_out, bias_out};
save_name = fullfile("Results", experiment_name, "First_withConductance.mat");
save(save_name,"fc_error", "outdata");
%(T,ave_fc,G,alpha, wgain, bgain,n_weigth, n_bias,params,opts, metric)
%[fc_error ,outdata] = gain_gridsearch_with_metrics(params.TMAX,ave_fc,G,alpha,wgain, bgain, n_weigth, n_bias,params,bo_opts, 'ssim'); % Optimizes FCD
%opt_res = load([checkoint_file]);
%[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')

%%
wgain = [1.6 2]; % Here only optimizing gain
bgain = [-0.1 0];
% Number of points in grid = n_weigth*n_bias
n_weigth = 40;
n_bias = 10;

%params.g_e = 1; % Conductance is replaced by heterogeneity experssion
%params.g_i = 1;



% Setting DMF parameters
N = size(params.C,1);
stren = sum(params.C);
isubfc = find(tril(ones(N),-1));
nsteps = params.TMAX.*(1000); % number of DMF timepoints
gamma_ent_fun = @(a) a(1) + log(a(2)) + log(gamma(a(1))) + (1-a(1))*psi(a(1));


opt_vars = [];


if length(wgain(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    wgainvals = optimizableVariable('wgain',[wgain(1) wgain(2)]);
    opt_vars = [wgainvals];
end

if length(bgain(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    bgainvals = optimizableVariable('bgain',[bgain(1) bgain(2)]);
    opt_vars = [opt_vars bgainvals];
end




weight_space = linspace(wgain(1), wgain(2), n_weigth);
bias_space = linspace(bgain(1), bgain(2), n_bias);
weight_range = 1:length(weight_space);
bias_range = 1:length(bias_space);
local_fc_error = ones(length(weight_space), length(bias_space));
local_fc = zeros(length(weight_space), length(bias_space), N, N);



parfor cur_weight=weight_range
    thispars=params;       
    %local_min = 1;    
    for cur_bias=bias_range        
        
        thispars.wgain = weight_space(cur_weight);                                           
        thispars.bgain = bias_space(cur_bias);                 
        % Simulating        
        [rates,bold] = DMF_hetero(thispars, nsteps,'both'); % runs simulation
        bold = bold(:,params.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        if isempty(bold)            
            local_fc_error(cur_weight, cur_bias) = nan;                    
        end
        % Filtering and computing FC
        filt_bold = filter_bold(bold',params.flp,params.fhi,params.TR);
        sim_fc = corrcoef(filt_bold);
        
        % Computing rates
        rates = rates(:,(params.burnout*1000):end);
        reg_fr = mean(rates,2);
        reg_ent = zeros(N,1);
        for n=1:N
            gamma_pars = gamfit(rates(n,:));
            reg_ent(n) = gamma_ent_fun(gamma_pars);
        end
    
        % Computing FC error: Mean Squared differences between vectorized FCs              
        mse_error= mean((sim_fc(isubfc)-ave_fc(isubfc)).^2); % MSE FC
        ssim_error = 1-ssim(ave_fc,sim_fc);
        corr_error = 1-corr2(ave_fc, sim_fc);        
        metrics=containers.Map;        
        metrics('ssim')=ssim_error;        
        
        %parsave(cur_weight,cur_bias, sim_fc, metrics(metric))
    
        local_fc_error(cur_weight, cur_bias)=metrics(metric);        
        local_fc(cur_weight, cur_bias, :, :) = sim_fc;
    
        
    end
end

[fc_error_bias, weight] = min(local_fc_error);
[fc_error, bias] = min(fc_error_bias);
weight_out = weight_space(weight(bias));                                           
bias_out = bias_space(bias); 
outdata = {squeeze(local_fc(weight(bias), bias, :, :)), weight_out, bias_out};
%(T,ave_fc,G,alpha, wgain, bgain,n_weigth, n_bias,params,opts, metric)
%[fc_error ,outdata] = gain_gridsearch_with_metrics(params.TMAX,ave_fc,G,alpha,wgain, bgain, n_weigth, n_bias,params,bo_opts, 'ssim'); % Optimizes FCD
%opt_res = load([checkoint_file]);
%[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
save_name = fullfile("Results", experiment_name, "Second_withConductance.mat");
save(save_name,"fc_error", "outdata");
%% Simulations
%
%[ params ] = DefaultParams('C',C);
%params.receptors = av/max(av);
stren = sum(params.C);
C = SC/max(max(SC))*0.2;
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.04; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = 0.07; % high cut-off of the bandpass filter 0.1
params.wsize = 30; % size of the FCD windows
params.overlap = 28; % overlap of the FCD windows
% IT IS TR=2 FOR ENZOS SLEEP DATA
params.TR = 2; % repetition time of the fMRI signal (will be used to simulate fMRI)
params.batch_size = 50000; % batch for
% circular buffer of simulation
% params.seed = 10; % initial condition for the simulation. 
% Heterogenity
params.receptors = av/max(av);
% Setting data constants
params.N=80;
params.NSUB=398;
params.TMAX=464;
params.START_TIME=101;

thispars = params;

%% SSIM
sub_experiment_name = 'Hetero_nogain_SSIM';
thispars.G = prev_res.best_pars_ssim.G;
%thispars.alpha = best_pars_ssim.alpha;
thispars.J = 0.75*thispars.G*stren' + 1; % updates it
thispars.wgain = 0;
thispars.bgain = 0;
% Run simulation for a given nb of steps (milliseconds)
nb_steps = 1500000;
parfor nsub=1:params.NSUB
    nsub
    BOLDNM = DMF_hetero(thispars, nb_steps);
    BOLDNM = filter_bold(BOLDNM', params.flp, params.fhi, ...
        params.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM5 = BOLDNM(:, 1+trans:end-trans);
    simulations5(:, :, nsub) = BOLDNM5(:, params.START_TIME:params.START_TIME+params.TMAX);
    simulationsFC5(:, :, nsub) = corrcoef(squeeze(simulations5(:, :, nsub))');
    trans = 20;
    BOLDNM20 = BOLDNM(:, 1+trans:end-trans);
    simulations20(:, :, nsub) = BOLDNM20(:, params.START_TIME:params.START_TIME+params.TMAX);
    simulationsFC20(:, :, nsub) = corrcoef(squeeze(simulations20(:, :, nsub))');
end
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "simulationsFC5","simulationsFC20", '-append')
h = figure();
sim_fc = mean(simulationsFC5 ,3);
disp(1-ssim(ave_fc, sim_fc))
imagesc(squeeze(mean(simulationsFC5 ,3)));
savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, "sim.fig"))

