clear;
close all;
basefold = './data/';
% time series
data_file = 'hcpunrelated100_REST_dkt68';
% structural connectivity
sc_file = 'SC_dtk68horn';
% heterogeneity to test
hetero_file = 'myelin_HCP_dk68';

load([basefold,data_file,'.mat'])
load([basefold,sc_file,'.mat'])
load([basefold,hetero_file,'.mat'])

% Set a name to save in a different folder
experiment_name = "baytrain_test_final";

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
stren = sum(C);
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
    % W refers to Wake data
    Wdata(:,:,nsub)=data(:,1:params.TMAX, nsub) ; % TS_W tiene registros de distinta longitud
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto

end

WFCdata = permute(WFCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);


opt_time_1 = 1200; % 20 min
opt_time_2 = 600; % 10 min

%opt_time_1 = 120; % 20 min
%opt_time_2 = 60; % 10 min


% If You Have A Saved Optimum Value For G and FIC Load It
%{
opt_mse = load('data/checkpoints/W_MSE_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat')
best_pars_mse = bestPoint(opt_mse.BayesoptResults,'Criterion','min-mean')
G = best_pars_mse.G; 
alpha = best_pars_mse.alpha; 
%}
% If not put the value that you want
params.G = 2.1;
% Alpha will be set to defualt

% receptors already set
params.scale = [0 2]; % Here only optimizing nm
params.bias = [-1 0];


if ~exist(fullfile("Figuras",experiment_name))
    mkdir(fullfile("Figuras",experiment_name))
end
if ~exist(fullfile("Results",experiment_name))
    mkdir(fullfile("Results",experiment_name))
end
   
if ~exist(fullfile("data/checkpoints",experiment_name))
    mkdir(fullfile("data/checkpoints",experiment_name))
end

checkpoint_folder = fullfile("data/checkpoints",experiment_name);

%
%% MSE
%
sub_experiment_name = "MSE";
filename = experiment_name+"-"+sub_experiment_name;
checkpoint_file = [fullfile(checkpoint_folder,filename+"_v1.mat")];
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Results",experiment_name, sub_experiment_name))
    mkdir(fullfile("Results",experiment_name, sub_experiment_name))
end


% Optimizing heterogenity parameters

bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkpoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_mse] = gain_fit_with_metrics(params.TMAX,emp_fc,[],params.G,params.alpha,params.scale,params.bias, params,bo_opts, 'mse'); % Optimizes FCD
opt_res = load([checkpoint_file]);
[best_pars_mse,est_min_ks_mse] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))

close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkpoint_file2 = [fullfile(checkpoint_folder,filename+"_v2.mat")];

bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective},...
        'SaveFileName',checkpoint_file2};

params.scale_finetune = [max(params.scale(1),best_pars_mse.scale-0.5) min(best_pars_mse.scale+0.5, params.scale(2))];
params.bias_finetune = [max(params.bias(1),best_pars_mse.bias-0.3) min(best_pars_mse.bias+0.3,params.bias(2))];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_mse] = gain_fit_with_metrics(params.TMAX,emp_fc,[],params.G,params.alpha,params.scale_finetune,params.bias_finetune,params,bo_opts2, 'mse'); 
best_pars_mse = bestPoint(bayesopt_out_mse, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "best_pars_mse", "bayesopt_out_mse", "params")
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;
%
%% SSIM
%

sub_experiment_name = "SSIM";
filename = experiment_name+"-"+sub_experiment_name;

if ~exist(fullfile("Results",experiment_name, sub_experiment_name))
    mkdir(fullfile("Results",experiment_name, sub_experiment_name))
end

if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name))
end
checkpoint_folder = fullfile("data/checkpoints",experiment_name);
checkpoint_file = [fullfile(checkpoint_folder,filename+"_v1.mat")];

% If You Have A Saved Optimum Value For G and FIC Load It
%{
opt_ssim = load('W_SSIM_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat');
best_pars_ssim = bestPoint(opt_ssim.BayesoptResults,'Criterion','min-mean');
G = best_pars_ssim.G; 
alpha = best_pars_ssim.alpha; 
%}
% G is set globally
% G = 2.1



bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkpoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = gain_fit_with_metrics(params.TMAX,emp_fc,[],params.G,params.alpha,params.scale,params.bias,params,bo_opts, 'ssim'); % Optimizes FCD
opt_res = load([checkpoint_file]);
[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')

% Not inputting to plot functions the folder. Just move from the default location
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))

close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkpoint_file2 = [fullfile(checkpoint_folder,filename+"_v2.mat")];
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective},...
        'SaveFileName',checkpoint_file2};

scale_finetune = [max(params.scale(1),best_pars_ssim.scale-0.5) min(best_pars_ssim.scale+0.5, params.scale(2))];
bias_finetune = [max(params.bias(1),best_pars_ssim.bias-0.3) min(best_pars_ssim.bias+0.3,params.bias(2))];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = gain_fit_with_metrics(params.TMAX,emp_fc,[],params.G,params.alpha,params.scale_finetune,params.bias_finetune,params,bo_opts2, 'ssim'); % Optimizes FCD
best_pars_ssim = bestPoint(bayesopt_out_ssim, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "best_pars_ssim", "bayesopt_out_ssim", "params")
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;
%
% CORR
%


sub_experiment_name = "CORR";
filename = experiment_name+"-"+sub_experiment_name;
checkpoint_file = [fullfile(checkpoint_folder,filename+"_v1.mat")];
if ~exist(fullfile("Results",experiment_name, sub_experiment_name))
    mkdir(fullfile("Results",experiment_name, sub_experiment_name))
end

if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name))
end
% If You Have A Saved Optimum Value For G and FIC Load It
%{
opt_corr = load('data/checkpoints/W_CORR_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat')
best_pars_corr = bestPoint(opt_corr.BayesoptResults,'Criterion','min-mean')
% Optimizable parameters
G = best_pars_corr.G; 
alpha = best_pars_corr.alpha; 
%}
% G is set globally
% G = 2.1


bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkpoint_file,...
        'PlotFcn', {@plot_mse,@plot_corr,@plot_ssim,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_corr] = gain_fit_with_metrics(params.TMAX,emp_fc,[],params.G,params.alpha,params.scale,params.bias,params,bo_opts, 'corr'); % Optimizes FCD
opt_res = load([checkpoint_file]);
[best_pars_corr,est_min_ks_corr] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))

close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkpoint_file2 = [fullfile(checkpoint_folder,filename+"_v2.mat")];
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective},...
        'SaveFileName',checkpoint_file2};

params.scale_finetune = [max(params.scale(1),best_pars_corr.scale-0.5) min(best_pars_corr.scale+0.5, params.scale(2))];
params.bias_finetune = [max(params.bias(1),best_pars_corr.bias-0.3) min(best_pars_corr.bias+0.3,params.bias(2))];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_corr] = gain_fit_with_metrics(params.TMAX,emp_fc,[],params.G,params.alpha,params.scale_finetune,params.bias_finetune,params,bo_opts2, 'corr'); % Optimizes FCD
best_pars_corr = bestPoint(bayesopt_out_corr, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "best_pars_corr", "bayesopt_out_corr", "params")

if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end

movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;


%% Simulations
% Use the same parameters as in the begining of the script
% I usually have to put like 10000000 more than params.*1000 to get the
% actual 464 seconds im simulating

nb_steps = 1500000;
n_simulations = 1:NSUBJECTS;

%%
%  MSE SIM
sub_experiment_name = "MSE";
filename = experiment_name+"-"+sub_experiment_name;
checkpoint_file = [fullfile(checkpoint_folder,filename+"_v2.mat")];
opt_mse = load(checkpoint_file);
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
params_train = load(save_name).params;
best_pars_mse = bestPoint(opt_mse.BayesoptResults,'Criterion','min-mean');
% We keep the same parameters then training, and replace the ones we tuned
params_train.scale = best_pars_mse.scale;
params_train.bias = best_pars_mse.bias;

% Run simulation for a given nb of steps (milliseconds)
for nsub=n_simulations
    nsub
    BOLDNM = DMF(params_train, nb_steps);
    BOLDNM = filter_bold(BOLDNM', params_train.flp, params_train.fhi, params_train.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:params_train.TMAX);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "simulationsFC", "-append")
h = figure();
sim_fc = mean(simulationsFC ,3);
imagesc(squeeze(sim_fc));
savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, sub_experiment_name+"_SIM.fig"))
disp("MSE error: " + mean((sim_fc(Isubdiag)-emp_fc(Isubdiag)).^2))

%% SSIM SIM
sub_experiment_name = "SSIM";
filename = experiment_name+"-"+sub_experiment_name;
checkpoint_file = [fullfile(checkpoint_folder,filename+"_v2.mat")];
opt_ssim = load(checkpoint_file);
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
params_train = load(save_name).params;
best_pars_ssim = bestPoint(opt_ssim.BayesoptResults,'Criterion','min-mean');

params_train.scale = best_pars_ssim.scale;
params_train.bias = best_pars_ssim.bias;


for nsub=n_simulations
    nsub
    BOLDNM = DMF_hetero(params_train, nb_steps);
    BOLDNM = filter_bold(BOLDNM', params_train.flp, params_train.fhi, params_train.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:params_train.TMAX);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end

%save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
%save(save_name, "simulationsFC")
h = figure();
sim_fc = mean(simulationsFC ,3);
imagesc(squeeze(sim_fc));
%savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, sub_experiment_name+"_SIM.fig"))
disp(1-ssim(emp_fc, sim_fc))

%% CORR SIM

sub_experiment_name = "CORR";
filename = experiment_name+"-"+sub_experiment_name;
checkpoint_file = [fullfile(checkpoint_folder,filename+"_v2.mat")];
opt_corr = load(checkpoint_file);
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
params_train = load(save_name).params;
best_pars_corr = bestPoint(opt_corr.BayesoptResults,'Criterion','min-mean');

params_train.scale = best_pars_corr.scale;
params_train.bias = best_pars_corr.bias;

  
for nsub=n_simulations
    nsub
    BOLDNM = DMF_hetero(params_train, nb_steps);
    BOLDNM = filter_bold(BOLDNM', params_train.flp, params_train.fhi, params_train.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:params_train.TMAX);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "simulationsFC", "-append")
h = figure();
sim_fc = mean(simulationsFC ,3);
imagesc(squeeze(sim_fc));
savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, sub_experiment_name+"_SIM.fig"))
disp(1-corr2(emp_fc, sim_fc))