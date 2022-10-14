clear all;
close all;
basefold = './data/';
data_file = 'ts_coma24_schaefer100';
sc_file = 'schaefer100_avg40subj';
hetero_file = 'myelin_HCP_dk68';
load([basefold,data_file,'.mat'])
load([basefold,sc_file,'.mat'])
load([basefold,hetero_file,'.mat'])
%%
experiment_name = 'Coma100';
sub_experiment_name = "Galpha_Control";
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
    data(:, :, nsub)=timeseries_CNT24{indexsub};
end

%% 

C = SC/max(max(SC))*0.2;
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
stren = sum(params.C);
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

params.NSIM=15;

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
params.G_range = [0.5 2.9];
params.alpha_range = [0.1 1];
%params.J = 0.75*params.G_range*stren' + 1;
params.scale = 0; % Here only optimizing gain
params.bias = 0;
nm_scale =0;
nm_bias = 0;
opt_time_1 = 7200; % 120 min
opt_time_2 = 3600; % 60 min
%opt_time_1 = 120; % 2 min
%opt_time_2 = 60; % 1 min


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

%
%% SSIM
%
checkoint_file = fullfile(basefold, checkpoint_folder, experiment_name, sub_experiment_name+"ssimv1.mat")
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = fit_with_metrics(params.TMAX,emp_fc,[],params.G_range,params.alpha_range,nm_scale, nm_bias,params,bo_opts, 'ssim'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))
close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = fullfile(basefold, checkpoint_folder, experiment_name, 'Finetune', sub_experiment_name+"ssimv2.mat")
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective},...
        'SaveFileName',checkoint_file2};

G_finetune = [max(params.G_range(1),best_pars_ssim.G-0.5) min(best_pars_ssim.G+0.5, params.G_range(2))];
alpha_finetune = [max(params.alpha_range(1),best_pars_ssim.alpha-0.15) min(best_pars_ssim.alpha+0.15, params.alpha_range(2))];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = fit_with_metrics(params.TMAX,emp_fc,[],G_finetune,alpha_finetune,nm_scale, nm_bias,params,bo_opts2, 'ssim'); % Optimizes FCD
best_pars_ssim = bestPoint(bayesopt_out_ssim, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, "SSIM_bay.mat");
save(save_name, "best_pars_ssim", "bayesopt_out_ssim")
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;
%


checkoint_file = fullfile(basefold, checkpoint_folder, experiment_name, sub_experiment_name+"msev1.mat")
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_mse] = fit_with_metrics(params.TMAX,emp_fc,[],params.G_range,params.alpha_range,nm_scale, nm_bias,params,bo_opts, 'mse'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_mse,est_min_ks_mse] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))
close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = fullfile(basefold, checkpoint_folder, experiment_name, 'Finetune', sub_experiment_name+"msev2.mat")
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_mse,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective},...
        'SaveFileName',checkoint_file2};

G_finetune = [max(params.G_range(1),best_pars_mse.G-0.5) min(best_pars_mse.G+0.5, params.G_range(2))];
alpha_finetune = [max(params.alpha_range(1),best_pars_mse.alpha-0.15) min(best_pars_mse.alpha+0.15, params.alpha_range(2))];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_mse] = fit_with_metrics(params.TMAX,emp_fc,[],G_finetune,alpha_finetune,nm_scale, nm_bias,params,bo_opts2, 'mse'); % Optimizes FCD
best_pars_mse = bestPoint(bayesopt_out_mse, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, "MSE_bay.mat");
save(save_name, "best_pars_mse", "bayesopt_out_mse")
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;
%


checkoint_file = fullfile(basefold, checkpoint_folder, experiment_name, sub_experiment_name+"corrv1.mat")
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_corr] = fit_with_metrics(params.TMAX,emp_fc,[],params.G_range,params.alpha_range,nm_scale, nm_bias,params,bo_opts, 'corr'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_corr,est_min_ks_corr] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))
close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = fullfile(basefold, checkpoint_folder, experiment_name, 'Finetune', sub_experiment_name+"corrv2.mat")
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_corr,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective},...
        'SaveFileName',checkoint_file2};

G_finetune = [max(params.G_range(1),best_pars_corr.G-0.5) min(best_pars_corr.G+0.5, params.G_range(2))];
alpha_finetune = [max(params.alpha_range(1),best_pars_corr.alpha-0.15) min(best_pars_corr.alpha+0.15, params.alpha_range(2))];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_corr] = fit_with_metrics(params.TMAX,emp_fc,[],G_finetune,alpha_finetune,nm_scale, nm_bias,params,bo_opts2, 'corr'); % Optimizes FCD
best_pars_corr = bestPoint(bayesopt_out_corr, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, "CORR_bay.mat");
save(save_name, "best_pars_corr", "bayesopt_out_corr")
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;
%


%% Simulations
%
%[ params ] = DefaultParams('C',C);
%params.receptors = av/max

thispars = params;
res = load('Results/Coma100/Galpha_MCS/SSIM_bay.mat');
best_pars_ssim = bestPoint(res.bayesopt_out_ssim, 'Criterion', 'min-mean');
%% SSIM
%sub_experiment_name = "BEI_gus_replicate";
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Results",experiment_name, sub_experiment_name))
    mkdir(fullfile("Results",experiment_name, sub_experiment_name))
end
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end

metric = 'ssim';
thispars.G = best_pars_ssim.G;  %prev_res.best_pars_mse.G;
thispars.alpha = best_pars_ssim.alpha;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it


% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000; %460800 = TMAX * TR * 1000
parfor nsub=1:thispars.NSUB
    nsub
    [rates, BOLDNM] = DMF(thispars, nb_steps, 'both');
    BOLDNM = filter_bold(BOLDNM', thispars.flp, thispars.fhi, thispars.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM5 = BOLDNM(:, 1+trans:end-trans);
    simulations5(:, :, nsub) = BOLDNM5(:, 1:thispars.TMAX);
    simulationsFC5(:, :, nsub) = corrcoef(squeeze(simulations5(:, :, nsub))');
    trans = 20;
end
save_name = fullfile("Results", experiment_name, sub_experiment_name, metric+"_simulations.mat");
save(save_name, "simulationsFC5", "thispars")
h = figure();
sim_fc = mean(simulationsFC5 ,3);
disp(1-ssim(emp_fc, sim_fc))
imagesc(squeeze(mean(simulationsFC5 ,3)));
savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, metric+"_sim.fig"))

