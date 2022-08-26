clear all;
close all;
basefold = './data/';
fcd_file = 'DataSleepW_N3'; % file with fc, fcd, fMRI and filter parameters for all subjects
load('data/heterogeneitys/myelin_aal.mat') % Heterogenity
load([basefold,fcd_file,'.mat'])

C = SC/max(max(SC))*0.2;
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.04; % low cut-off of the bandpass filter 0.01 for aal N3
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
params.receptors(find(params.receptors==0))=mean(params.receptors);
% Setting data constants
params.N=90;
params.NSUB=15;
params.TMAX=198;
Isubdiag = find(tril(ones(params.N),-1));
isubfc = Isubdiag;

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;
    Wdata(:,:,nsub)=TS_W{1,nsub}(:,1:params.TMAX) ; % TS_W tiene registros de distinta longitud
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto

end

for nsub=indexsub
    nsub;
    N3data(:,:,nsub)=TS_N3{1,nsub}(:,1:params.TMAX) ; % TS_W tiene registros de distinta longitud
    N3dataF(:,:,nsub) = permute(filter_bold(N3data(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    N3FCdata(nsub,:,:)=corrcoef(squeeze(N3data(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    NFCdataF(nsub,:,:)=corrcoef(squeeze(N3dataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end
WFCdata = permute(WFCdata, [2,3,1]);
N3FCdata = permute(N3FCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
N3FCdataF = permute(NFCdataF, [2,3,1]);

ave_fc = mean(N3FCdataF,3);
%ave_fc = mean(WFCdataF,3);





%%
% MSE
%
opt_mse = load('data/checkpoints/N3_MSE_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat')
best_pars_mse = bestPoint(opt_mse.BayesoptResults,'Criterion','min-mean')
% Optimizable parameters
G = best_pars_mse.G; 
alpha = best_pars_mse.alpha; 
% receptors already set
nm = [-4 4]; % Here only optimizing nm
nm_bias = [-4 4];
T = params.TMAX; % seconds to simulate, should be comparable to the empirical recording time (~7-10 minutes of resting state)
% Optimizing heterogenity parameters

checkpoint_folder = 'checkpoints/';
experiment_name = 'N3_MSE_FIC';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_v1.mat'];
opt_time_1 = 1200; % 20 min
opt_time_2 = 600; % 10 min
%opt_time_1 = 120; % 2 min
%opt_time_2 = 60; % 1 min


bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_mse] = fit_with_metrics(T,ave_fc,[],G,alpha_slope,nm, nm_bias,params,bo_opts, 'mse'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_mse,est_min_ks_mse] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')

mkdir Figuras/N3_FC_FIC/ MSE/
movefile Figuras/*plot.fig Figuras/N3_FC_FIC/MSE/
close all;
%% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = [strcat(basefold, checkpoint_folder),strcat(experiment_name, 'finetune'),'_checkpoint_dmf_bayesopt_N',num2str(90),'_v2.mat'];
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

nm = [max(-4,best_pars_mse.nm-0.5) min(best_pars_mse.nm+0.5, 4)];
nm_bias = [max(-2,best_pars_mse.nm_bias-0.3) min(best_pars_mse.nm_bias+0.3, 2)];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_mse] = fit_with_metrics(T,ave_fc,[],G,alpha_slope,nm, nm_bias,params,bo_opts2, 'mse'); 

mkdir Figuras/N3_FC_FIC/MSE/ Finetune/
movefile Figuras/*plot.fig Figuras/N3_FC_FIC/MSE/Finetune
close all;
%%
% SSIM
%

opt_ssim = load('data/checkpoints/N3_SSIM_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat')
best_pars_ssim = bestPoint(opt_ssim.BayesoptResults,'Criterion','min-mean')
% Optimizable parameters
G = best_pars_ssim.G; 
alpha_slope = best_pars_ssim.alpha; 

nm = [-4 4]; % Here only optimizing nm
nm_bias = [-4 4];

experiment_name = 'N3_SSIM_FIC';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_v1.mat'];
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = fit_with_metrics(T,ave_fc,[],G,alpha_slope,nm, nm_bias,params,bo_opts, 'ssim'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')

mkdir Figuras/N3_FC_FIC/ SSIM/
movefile Figuras/*_plot.fig Figuras/N3_FC_FIC/SSIM/
close all;
%% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = [strcat(basefold, checkpoint_folder),strcat(experiment_name, 'finetune'),'_checkpoint_dmf_bayesopt_N',num2str(90),'_v2.mat'];
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

nm = [max(-4,best_pars_ssim.nm-0.5) min(best_pars_ssim.nm+0.5, 4)];
nm_bias = [max(-4,best_pars_ssim.nm_bias-0.3) min(best_pars_ssim.nm_bias+0.3, 4)];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = fit_with_metrics(T,ave_fc,[],G,alpha_slope,nm, nm_bias,params,bo_opts2, 'ssim'); % Optimizes FCD

mkdir Figuras/N3_FC_FIC/SSIM/ Finetune/
movefile Figuras/*_plot.fig Figuras/N3_FC_FIC/SSIM/Finetune
close all;
%%
% CORR
%

opt_corr = load('data/checkpoints/N3_CORR_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat')
best_pars_corr = bestPoint(opt_corr.BayesoptResults,'Criterion','min-mean')
% Optimizable parameters
G = best_pars_corr.G; 
alpha_slope = best_pars_corr.alpha; 

nm = [-4 4]; % Here only optimizing nm
nm_bias = [-4 4];

experiment_name = 'N3_CORR_FIC';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_v1.mat'];
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_corr,@plot_ssim,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_corr] = fit_with_metrics(T,ave_fc,[],G,alpha_slope,nm, nm_bias,params,bo_opts, 'corr'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_corr,est_min_ks_corr] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
mkdir Figuras/N3_FC_FIC/ CORR/
movefile Figuras/*_plot.fig Figuras/N3_FC_FIC/CORR/
close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = [strcat(basefold, checkpoint_folder),strcat(experiment_name, 'finetune'),'_checkpoint_dmf_bayesopt_N',num2str(90),'_v2.mat'];
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

nm = [max(-4,best_pars_corr.nm-0.5) min(best_pars_corr.nm+0.5, 4)];
nm_bias = [max(-4,best_pars_corr.nm_bias-0.3) min(best_pars_corr.nm_bias+0.3, 4)];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_corr] = fit_with_metrics(T,ave_fc,[],G,alpha,nm, nm_bias,params,bo_opts2, 'corr'); % Optimizes FCD

mkdir Figuras/N3_FC_FIC/CORR/ Finetune/
movefile Figuras/*_plot.fig Figuras/N3_FC_FIC/CORR/Finetune
close all;

% Simulations
%%
params = DefaultParams();
params.receptors = av/max(av);
params.receptors(find(params.receptors==0))=mean(params.receptors);
stren = sum(params.C);
thispars = params;

thispars.G = 0.9721;
thispars.alpha = 0.2422;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it
fic_nm = thispars.receptors.*best_pars_mse.nm + best_pars_mse.nm_bias; % Could add bias
thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC

% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000;
for nsub=1:15
    nsub
    BOLDNM = DMF(thispars, nb_steps);
    BOLDNM = filter_bold(BOLDNM', params.flp, params.fhi, params.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:198);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end
save("Results/N3_MSE_FIC.mat", "simulationsFC", "best_pars_mse", "bayesopt_out_mse")
h = figure();
imagesc(squeeze(mean(simulationsFC ,3)));
savefig(h, "Figuras/N3_FC_FIC/MSE/MSE_sim.fig")
%%
thispars.G = 1.822;
thispars.alpha = 0.692;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it
optssim = load('data/checkpoints/N3_SSIM_FICfinetune_checkpoint_dmf_bayesopt_N90_v2.mat');
best_pars_ssim = bestPoint(optssim.BayesoptResults, 'Criterion', 'min-mean');
fic_nm = thispars.receptors.*best_pars_ssim.nm + best_pars_ssim.nm_bias; % Could add bias
thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC
% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000;
for nsub=1:15
    nsub
    BOLDNM = DMF(thispars, nb_steps);
    BOLDNM = filter_bold(BOLDNM', thispars.flp, thispars.fhi, thispars.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:198);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end
save("Results/N3_SSIM_FIC.mat", "simulationsFC", "best_pars_ssim", "bayesopt_out_ssim")
h = figure();
imagesc(squeeze(mean(simulationsFC ,3)));
savefig(h, "Figuras/N3_FC_FIC/SSIM/SSIM_sim.fig")
%%
thispars.G = best_pars_corr.G;
thispars.alpha = best_pars_corr.alpha;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it

% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000;
  
for nsub=1:15
    nsub
    BOLDNM = DMF(thispars, nb_steps);
    BOLDNM = filter_bold(BOLDNM', params.flp, params.fhi, params.TR);
    BOLDNM = BOLDNM';   
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:198);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end
save("Results/N3_CORR_FIC.mat", "simulationsFC", "best_pars_corr", "bayesopt_out_corr")
h = figure();
imagesc(squeeze(mean(simulationsFC ,3)));
savefig(h, "Figuras/N3_FC_FIC/CORR/CORR_sim.fig")