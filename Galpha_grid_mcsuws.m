clear all;
close all;
basefold = './data/';
data_file = 'ts_coma24_schaefer100';
sc_file = 'schaefer100_avg40subj';
hetero_file = 'myelin_HCP_dk68';
load([basefold,data_file,'.mat'])
load([basefold,sc_file,'.mat'])
load([basefold,hetero_file,'.mat'])

experiment_name = 'Coma100';
sub_experiment_name = "Galpha_Grid_MCS";
% Dataset values
TR = 2.4; %This is for GusDeco at dkt68
NSUBJECTS = 13;
NREGIONS = 100;
MAXTIME = 192;
% SC is SC
RECEPTORS = 0;

% Save in data the timeseries
indexsub=1:NSUBJECTS;
for nsub=indexsub
    data(:, :, nsub)=timeseries_MCS24{indexsub};
end

%

C = SC/max(max(SC))*0.2;
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.04; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = 0.07; % high cut-off of the bandpass filter 0.1
params.wsize = 30; % size of the FCD windows
params.overlap = 28; % overlap of the FCD windows
params.TR = TR; % repetition time of the fMRI signal (will be used to simulate fMRI)
% Setting data constants
params.receptors = RECEPTORS;
params.N=NREGIONS;
params.NSUB=NSUBJECTS;
params.TMAX=MAXTIME;

params.NSIM=12;

Isubdiag = find(tril(ones(params.N),-1));

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=data(:, 1:params.TMAX, nsub); 
    Wdata(:, :, nsub) = detrend(Wdata(:, :, nsub));
    Wdata(:, :, nsub) = demean( Wdata(:, :, nsub));
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end

WFCdata = permute(WFCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);

% Optimizable parameters
% Setting DMF parameters
N = size(params.C,1);
stren = sum(params.C);
isubfc = find(tril(ones(N),-1));

nsteps = params.TMAX.*(1000); % number of DMF timepoints

gamma_ent_fun = @(a) a(1) + log(a(2)) + log(gamma(a(1))) + (1-a(1))*psi(a(1));
params.G = [0.5 2.9];
params.alpha = [0.1 1];
%params.J = 0.75*params.G*stren' + 1;
params.scale = 0;%[0 2]; % Here only optimizing gain
params.bias = 0;%[-0.9 0];
% Number of points in grid = n_weigth*n_bias
n_weigth = 45;
n_bias = 18;


checkpoint_folder = 'checkpoints/';

if ~exist(fullfile("Figuras",experiment_name))
    mkdir(fullfile("Figuras",experiment_name))
end
if ~exist(fullfile("Results",experiment_name))
    mkdir(fullfile("Results",experiment_name))
end
   
if ~exist(fullfile("data/checkpoints",experiment_name))
    mkdir(fullfile("data/checkpoints",experiment_name))
end


if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Results",experiment_name, sub_experiment_name))
    mkdir(fullfile("Results",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end

% Actual parameter
weight_space = linspace(params.G(1), params.G(2), n_weigth);
bias_space = linspace(params.alpha(1), params.alpha(2), n_bias);
weight_range = 1:n_weigth;
bias_range = 1:n_bias;
% Here we'll keep a matrix with the errors for all parameter values
local_fc_error = ones(n_weigth, n_bias);
local_fc_error_mse = ones(n_weigth, n_bias);
local_fc_error_corr = ones(n_weigth, n_bias);
local_fc = zeros(n_weigth, n_bias, N, N);

%
for cur_weight=weight_range    
    
    for cur_bias=bias_range   
        %bias_space(cur_bias);
        thispars=params;  
        thispars.G = weight_space(cur_weight);                                           
        thispars.alpha = bias_space(cur_bias);       
        thispars.J = thispars.alpha*thispars.G*stren' + 1;
        % Simulating  
        simulations=zeros(N,N,params.NSIM);                       
        parfor nsim=1:params.NSIM                         
            [rates,bold] = DMF_hetero(thispars, nsteps,'both'); % runs simulation
            bold = bold(:,params.burnout:end); % remove initial transient
            bold(isnan(bold))=0;
            bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));            
            if isempty(bold)          
                fprintf("%d - %d Gave nan\n",cur_weight, cur_bias)
                %local_fc_error(cur_weight, cur_bias) = nan;                    
                %local_fc_error_mse(cur_weight, cur_bias) = nan;   
                %local_fc_error_corr(cur_weight, cur_bias) = nan;   
            end            
            % Filtering and computing FC
            filt_bold = filter_bold(bold',params.flp,params.fhi,params.TR);
            sim_fc = corrcoef(filt_bold);  
            simulations(:, :, nsim) = sim_fc;
        end
        % Not Computing rates for this stage
        %rates = rates(:,(params.burnout*1000):end);
        %reg_fr = mean(rates,2);
        %reg_ent = zeros(N,1);
        %for n=1:N
        %    gamma_pars = gamfit(rates(n,:));
        %    reg_ent(n) = gamma_ent_fun(gamma_pars);
        %end
        sim_fc = mean(simulations,3);
        % Computing FC error. Default is SSIM
        mse_error= mean((sim_fc(isubfc)-emp_fc(isubfc)).^2); % MSE FC
        ssim_error = 1-ssim(emp_fc,sim_fc);
        corr_error = 1-corr2(emp_fc, sim_fc);        
        local_fc_error(cur_weight, cur_bias)=ssim_error;        
        local_fc_error_corr(cur_weight, cur_bias)=corr_error;
        local_fc_error_mse(cur_weight, cur_bias)=mse_error;
        local_fc(cur_weight, cur_bias, :, :) = sim_fc;
               
    end       
   
end
disp("Finished 1nd run ok")
[fc_error_bias, weight_min] = min(local_fc_error);
[fc_error, bias_min] = min(fc_error_bias);
weight_out = weight_space(weight_min(bias_min));                                           
bias_out = bias_space(bias_min); 
% See to add local_fc to outdata
outdata = {squeeze(local_fc(weight_min(bias_min), bias_min, :, :)), weight_out, bias_out};
save_name = fullfile("Results", experiment_name,sub_experiment_name,  "Results.mat");
save(save_name,"local_fc", "outdata", "local_fc_error", "local_fc_error_mse", "local_fc_error_corr", "params");
%
clear all;
close all;
basefold = './data/';
data_file = 'ts_coma24_schaefer100';
sc_file = 'schaefer100_avg40subj';
hetero_file = 'myelin_HCP_dk68';
load([basefold,data_file,'.mat'])
load([basefold,sc_file,'.mat'])
load([basefold,hetero_file,'.mat'])

experiment_name = 'Coma100';
sub_experiment_name = "Galpha_Grid_UWS";
% Dataset values
TR = 2.4; %This is for GusDeco at dkt68
NSUBJECTS = 13;
NREGIONS = 100;
MAXTIME = 192;
% SC is SC
RECEPTORS = 0;

% Save in data the timeseries
indexsub=1:NSUBJECTS;
for nsub=indexsub
    data(:, :, nsub)=timeseries_UWS24{indexsub};
end

%

C = SC/max(max(SC))*0.2;
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.04; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = 0.07; % high cut-off of the bandpass filter 0.1
params.wsize = 30; % size of the FCD windows
params.overlap = 28; % overlap of the FCD windows
params.TR = TR; % repetition time of the fMRI signal (will be used to simulate fMRI)
% Setting data constants
params.receptors = RECEPTORS;
params.N=NREGIONS;
params.NSUB=NSUBJECTS;
params.TMAX=MAXTIME;

params.NSIM=12;

Isubdiag = find(tril(ones(params.N),-1));

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=data(:, 1:params.TMAX, nsub); 
    Wdata(:, :, nsub) = detrend(Wdata(:, :, nsub));
    Wdata(:, :, nsub) = demean( Wdata(:, :, nsub));
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end

WFCdata = permute(WFCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);

% Optimizable parameters
% Setting DMF parameters
N = size(params.C,1);
stren = sum(params.C);
isubfc = find(tril(ones(N),-1));

nsteps = params.TMAX.*(1000); % number of DMF timepoints

gamma_ent_fun = @(a) a(1) + log(a(2)) + log(gamma(a(1))) + (1-a(1))*psi(a(1));
params.G = [0.5 2.9];
params.alpha = [0.1 1];
%params.J = 0.75*params.G*stren' + 1;
params.scale = 0;%[0 2]; % Here only optimizing gain
params.bias = 0;%[-0.9 0];
% Number of points in grid = n_weigth*n_bias
n_weigth = 45;
n_bias = 18;


checkpoint_folder = 'checkpoints/';

if ~exist(fullfile("Figuras",experiment_name))
    mkdir(fullfile("Figuras",experiment_name))
end
if ~exist(fullfile("Results",experiment_name))
    mkdir(fullfile("Results",experiment_name))
end
   
if ~exist(fullfile("data/checkpoints",experiment_name))
    mkdir(fullfile("data/checkpoints",experiment_name))
end


if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Results",experiment_name, sub_experiment_name))
    mkdir(fullfile("Results",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end

% Actual parameter
weight_space = linspace(params.G(1), params.G(2), n_weigth);
bias_space = linspace(params.alpha(1), params.alpha(2), n_bias);
weight_range = 1:n_weigth;
bias_range = 1:n_bias;
% Here we'll keep a matrix with the errors for all parameter values
local_fc_error = ones(n_weigth, n_bias);
local_fc_error_mse = ones(n_weigth, n_bias);
local_fc_error_corr = ones(n_weigth, n_bias);
local_fc = zeros(n_weigth, n_bias, N, N);


for cur_weight=weight_range    
    
    for cur_bias=bias_range   
        %bias_space(cur_bias);
        thispars=params;  
        thispars.G = weight_space(cur_weight);                                           
        thispars.alpha = bias_space(cur_bias);       
        thispars.J = thispars.alpha*thispars.G*stren' + 1;
        % Simulating  
        simulations=zeros(N,N,params.NSIM);                       
        parfor nsim=1:params.NSIM                         
            [rates,bold] = DMF_hetero(thispars, nsteps,'both'); % runs simulation
            bold = bold(:,params.burnout:end); % remove initial transient
            bold(isnan(bold))=0;
            bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));            
            if isempty(bold)          
                fprintf("%d - %d Gave nan\n",cur_weight, cur_bias)
                %local_fc_error(cur_weight, cur_bias) = nan;                    
                %local_fc_error_mse(cur_weight, cur_bias) = nan;   
                %local_fc_error_corr(cur_weight, cur_bias) = nan;   
            end            
            % Filtering and computing FC
            filt_bold = filter_bold(bold',params.flp,params.fhi,params.TR);
            sim_fc = corrcoef(filt_bold);  
            simulations(:, :, nsim) = sim_fc;
        end
        % Not Computing rates for this stage
        %rates = rates(:,(params.burnout*1000):end);
        %reg_fr = mean(rates,2);
        %reg_ent = zeros(N,1);
        %for n=1:N
        %    gamma_pars = gamfit(rates(n,:));
        %    reg_ent(n) = gamma_ent_fun(gamma_pars);
        %end
        sim_fc = mean(simulations,3);
        % Computing FC error. Default is SSIM
        mse_error= mean((sim_fc(isubfc)-emp_fc(isubfc)).^2); % MSE FC
        ssim_error = 1-ssim(emp_fc,sim_fc);
        corr_error = 1-corr2(emp_fc, sim_fc);        
        local_fc_error(cur_weight, cur_bias)=ssim_error;        
        local_fc_error_corr(cur_weight, cur_bias)=corr_error;
        local_fc_error_mse(cur_weight, cur_bias)=mse_error;
        local_fc(cur_weight, cur_bias, :, :) = sim_fc;
               
    end       
   
end
disp("Finished 1nd run ok")
[fc_error_bias, weight_min] = min(local_fc_error);
[fc_error, bias_min] = min(fc_error_bias);
weight_out = weight_space(weight_min(bias_min));                                           
bias_out = bias_space(bias_min); 
% See to add local_fc to outdata
outdata = {squeeze(local_fc(weight_min(bias_min), bias_min, :, :)), weight_out, bias_out};
save_name = fullfile("Results", experiment_name,sub_experiment_name,  "Results.mat");
save(save_name,"local_fc", "outdata", "local_fc_error", "local_fc_error_mse", "local_fc_error_corr", "params");