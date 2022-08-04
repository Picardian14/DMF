%% Workflow for fitting DMF with Dynamic FIC / Load data
% Uses either Functional Connectivity matrix or Functional Connectivity
% Dynamics matrix to fit the DMF using Bayesian Optimization
% FC is a N x N x nsubs matrix and FCD is a nwins x nwins x nsubs matrix,
% where 'N' is the number of brain regions, 'nwins' the number of windows used
% to compute the FCD, and 'nsubs' the number of recorded subjects.
% tr, flp, and fhi corresponds to the repetition time of the fMRI
% recordings, the low. and high- pass frequency of the bandpass filter of
% fMRI signals.

N=80;
NPARCELLS = N;
NSUB=100;
NSUBSIM=15; % cantidad de simulaciones a hacer, suele ser igual a la cantidad de sujetos
Tmax=1200;
indexsub=1:NSUB;
Isubdiag = find(tril(ones(N),-1));

basefold = './data/';
fcd_file = 'hcp1003_REST1_LR_dbs80'; % file with fc, fcd, fMRI and filter parameters for all subjects
load([basefold,fcd_file,'.mat'])
load([basefold, 'aal90_fc_fcd_wake.mat'])
load([basefold,'SC_dbs80HARDIFULL.mat']) % structural connectivity matrix
%% 

C = SC_dbs80HARDI./max(SC_dbs80HARDI(:)).*0.2;

%Se toma para todos la misma cantidad de puntos en el tiempo: Tmax. De paso se sacan las correlaciones para todos los nodos 
for nsub=indexsub
    nsub
    tsdata(:, :, nsub) = subject{1,nsub}.dbs80ts(:, 1:Tmax) ; % TS_W tiene registros de distinta longitud
    FCdata_t(nsub, :, :) = corrcoef(squeeze(tsdata(:, :, nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end
FCdata = permute(FCdata_t, [2,3,1]);

nsubs = NSUB;
N = length(C);
% SIN FCD POR AHORA
%nwins = size(fcd,1);
ave_fc = mean(FCdata,3);

%% Preparing parameters
% dmf parameters
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = flp; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = fhi; % high cut-off of the bandpass filter 0.1
params.wsize = wsize; % size of the FCD windows
params.overlap = overlap; % overlap of the FCD windows
params.TR = tr; % repetition time of the fMRI signal (will be used to simulate fMRI)
params.batch_size = 50000; % batch for circular buffer of simulation
% params.seed = 10; % initial condition for the simulation. 

%% Running optimizer
% Bayes optimization options
% The optimizer runs for a very long time, but it saves in disk the output
% of each iteration, so it can be cancelled and resumed then reading the
% files from disk.
opt_time = 3600*7*24; % a week
checkoint_file = [basefold,'HCP_100sub','_checkpoint_dmf_bayesopt_N',num2str(N),'_v1.mat'];
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file};


% Optimizable parameters
G = [1 2]; % Here only optimizing G
alpha_slope = 0.75; % fix alpha-> determines the local inhibitiory feedback. Can be optimized as well.
% if only FC is provided (i.e. fcd = []) only optimized with the FC.
% If FCD is provided, optimizes using FCD and also computes the respective
% goodness of fit to the FC.

% Use short time like 300 to test. If not, 600 for a final optimization
% value
T = 600; % seconds to simulate, should be comparable to the empirical recording time (~7-10 minutes of resting state)
% FC
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,[], ...
    G,alpha_slope,params,bo_opts); % Optimizes FC
% or FCD
%[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,fcd,G,alpha_slope,params,bo_opts); % Optimizes FCD

%% Loading optimization results from disk and resuming optimization
% Assuming training is done, and we have a checkpoint file
opt_res = load(checkoint_file);
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;

% Bayes optimization options
checkoint_file2 = [basefold,'HCP','_checkpoint_dmf_bayesopt_N',num2str(N),'_v2.mat'];
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time   ,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file2};
    
% Optimizable parameters
G = [1 2];
alpha_slope = 0.75; % fix alpha

T = 1200; % seconds
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] =  fit_fc_fcd_dmf_only_slope(T,ave_fc,[],G,alpha_slope,params,bo_opts2); % optimizes FC
%[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,fcd,G,alpha_slope,params,bo_opts2); % optimizes FCD

%% Extracting Optimal Parameters
opt_res2 = load(checkoint_file2);
[best_pars,est_min_ks] = bestPoint(opt_res2.BayesoptResults,'Criterion','min-mean')

%% Optimizing alpha_slope of the DMF with fix G
selG = 2.4;
checkoint_file3 = [basefold,fcd_file,'_checkpoint_dmf_bayesopt_N',num2str(N),'_G_',num2str(selG),'_alpha_v1.mat'];
bo_opts3 = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',8,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',3600*7*24   ,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file3};


% Optimizable parameters
G = 2.4; % fix G from pervious optimizations
alpha_slope = [0.7 0.8]; % free alpha, tuning FIC

T = 600; % seconds
% [opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,[],G,alpha_slope,params,bo_opts3); % optimizes fc
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,fcd,G,alpha_slope,params,bo_opts3); % optimizes fcd


%% Optimizing G and alpha_slope
checkoint_file4 = [basefold,fcd_file,'_checkpoint_dmf_bayesopt_N',num2str(N),'_G_and_alpha_v1.mat'];
bo_opts4 = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',8,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',3600*7*24   ,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file4};


% Optimizable parameters
G = [0 3]; % free G
alpha_slope = [0.7 0.8]; % free alpha, tuning FIC

T = 600; % seconds
% [opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,[],G,alpha_slope,params,bo_opts4); % fc
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,fcd,G,alpha_slope,params,bo_opts4); % fcd



