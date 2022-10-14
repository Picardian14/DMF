clear all;
close all;
basefold = './data/';
data_file = 'hcpunrelated100_REST_dkt68';
sc_file = 'SC_dtk68horn';
hetero_file = 'myelin_HCP_dk68';
load([basefold,data_file,'.mat'])
load([basefold,sc_file,'.mat'])
load([basefold,hetero_file,'.mat'])

experiment_name = 'HCP68';
sub_experiment_name = "G_gridsearch";
% Dataset values
TR = 0.72; %This is for GusDeco at dkt68
NSUBJECTS = 100;
NREGIONS = 68;
MAXTIME = 464;
SC = Cnew;
RECEPTORS = t1t2Cortex;
RECEPTORS = RECEPTORS/max(RECEPTORS)-min(RECEPTORS);
RECEPTORS = RECEPTORS - max(RECEPTORS) + 1;

% Save in data the timeseries
indexsub=1:NSUBJECTS;
for nsub=indexsub
    data(:, :, nsub)=subject{nsub}.dkt68ts(:, :);
end


%% 

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
Isubdiag = find(tril(ones(params.N),-1));

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=data(:, 1:params.TMAX, nsub) ; 
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
params.G = [0.5 3];%2.1;
%params.J = 0.75*params.G*stren' + 1;
scale = 0;%[0 2]; % Here only optimizing gain
bias = 0;%[-1 0];
% Number of points in grid = n_weigth*n_bias
n_weigth = 40;
n_bias = 20;
n_g = 25;

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

g_space = linspace(params.G(1), params.G(2), n_g);
g_range = 1:n_g;
% Here we'll keep a matrix with the errors for all parameter values
%{
local_fc_error = ones(n_weigth, n_bias);
local_fc_error_mse = ones(n_weigth, n_bias);
local_fc_error_corr = ones(n_weigth, n_bias);
local_fc = zeros(n_weigth, n_bias, N, N);
%}
local_fc_error = ones(n_g);
local_fc_error_mse = ones(n_g);
local_fc_error_corr = ones(n_g);
local_fc = zeros(n_g, N, N);
%%

parfor cur_weight=g_range
    thispars=params;       
    cur_weight
    thispars.G = cur_weight;
    thispars.J = 0.75*thispars.G*stren' + 1;
    thispars.scale = 0;%weight_space(cur_weight);                                           
    thispars.bias = 0;%bias_space(cur_bias);                 
    % Simulating  
    [rates,bold] = DMF_hetero(thispars, nsteps,'both'); % runs simulation
    bold = bold(:,params.burnout:end); % remove initial transient
    bold(isnan(bold))=0;
    bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
    if isempty(bold)            
        %{
        local_fc_error(cur_weight, cur_bias) = nan;                    
        local_fc_error_mse(cur_weight, cur_bias) = nan;   
        local_fc_error_corr(cur_weight, cur_bias) = nan;   
        %}
        local_fc_error(cur_weight) = nan;                    
        local_fc_error_mse(cur_weight) = nan;   
        local_fc_error_corr(cur_weight) = nan;   
    end
    % Filtering and computing FC
    filt_bold = filter_bold(bold',params.flp,params.fhi,params.TR);
    sim_fc = corrcoef(filt_bold);        
    % Not Computing rates for this stage
    %rates = rates(:,(params.burnout*1000):end);
    %reg_fr = mean(rates,2);
    %reg_ent = zeros(N,1);
    %for n=1:N
    %    gamma_pars = gamfit(rates(n,:));
    %    reg_ent(n) = gamma_ent_fun(gamma_pars);
    %end

    % Computing FC error. Default is SSIM
    mse_error= mean((sim_fc(isubfc)-emp_fc(isubfc)).^2); % MSE FC
    ssim_error = 1-ssim(emp_fc,sim_fc);
    corr_error = 1-corr2(emp_fc, sim_fc);        
            

    local_fc_error(cur_weight)=ssim_error;        
    local_fc_error_corr(cur_weight)=corr_error;
    local_fc_error_mse(cur_weight)=mse_error;
    local_fc(cur_weight, :, :) = sim_fc;

    

end
disp("Finished 1st run ok")
[fc_error_bias, weight_min] = min(local_fc_error);

weight_out = g_space(weight_min);                                           

% See to add local_fc to outdata
outdata = {squeeze(local_fc(weight_min, :, :)), weight_out};
save_name = fullfile("Results", experiment_name,sub_experiment_name,  "Results.mat");
save(save_name,"local_fc", "outdata", "local_fc_error", "local_fc_error_mse", "local_fc_error_corr", "params");
%%
h = figure(); 
imagesc(local_fc_error_mse); 
xticks(bias_range);
yticks(weight_range); 
xticklabels(round(bias_space,2)); 
yticklabels(round(weight_space,2))
save_fig = fullfile("Figuras/", experiment_name, sub_experiment_name, "MSE_Map.fig");
savefig(save_fig, h);



%% Simulations
%
ssim_w = 1.5385;
ssim_b = -0.1053;
mse_w = 1.8974;
mse_b = -0.4211;
corr_w = 0.8205;
corr_b = -0.5789;
thispars = params;

%% SSIM
%sub_experiment_name = 'Hetero_nogain_SSIM';
thispars.G = 2.1;
%thispars.alpha = best_pars_ssim.alpha;
thispars.J = 0.75*thispars.G*stren' + 1; % updates it
thispars.scale = ssim_w;
thispars.bias = ssim_b;
% Run simulation for a given nb of steps (milliseconds) Should repeat the
% same amount in sim than in train
nb_steps = 1500000;
parfor nsub=1:params.NSUB
    nsub
    BOLDNM = DMF_hetero(thispars, nb_steps);
    BOLDNM = filter_bold(BOLDNM', thispars.flp, thispars.fhi, ...
        thispars.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM5 = BOLDNM(:, 1+trans:end-trans);
    simulations5(:, :, nsub) = BOLDNM5(:, 1:thispars.TMAX);
    simulationsFC5(:, :, nsub) = corrcoef(squeeze(simulations5(:, :, nsub))');
    trans = 20;
    BOLDNM20 = BOLDNM(:, 1+trans:end-trans);
    simulations20(:, :, nsub) = BOLDNM20(:, 1:thispars.TMAX);
    simulationsFC20(:, :, nsub) = corrcoef(squeeze(simulations20(:, :, nsub))');
end
save_name = fullfile("Results", experiment_name, sub_experiment_name, "SimulationSSIM.mat");
save(save_name, "simulationsFC5","simulationsFC20", "thispars")
h = figure();
sim_fc = mean(simulationsFC5 ,3);
disp(1-ssim(emp_fc, sim_fc))
imagesc(squeeze(sim_fc));
savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, "ssim_sim.fig"))

