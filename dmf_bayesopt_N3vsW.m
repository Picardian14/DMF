%% Workflow for fitting DMF with Dynamic FIC / Load data
% Uses either Functional Connectivity matrix or Functional Connectivity
% Dynamics matrix to fit the DMF using Bayesian Optimization
% FC is a N x N x nsubs matrix and FCD is a nwins x nwins x nsubs matrix,
% where 'N' is the number of brain regions, 'nwins' the number of windows used
% to compute the FCD, and 'nsubs' the number of recorded subjects.
% tr, flp, and fhi corresponds to the repetition time of the fMRI
% recordings, the low. and high- pass frequency of the bandpass filter of
% fMRI signals.
clear all;
close all;
basefold = './data/';
fcd_file = 'DataSleepW_N3'; % file with fc, fcd, fMRI and filter parameters for all subjects
load('data/heterogeneitys/myelin_aal.mat') % Heterogenity
load([basefold,fcd_file,'.mat'])
C = SC/max(max(SC))*0.2;
%% Preparing parameters
% dmf parameters
[ params ] = DefaultParams('C',C); % creates default parameters for the simulation
params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.01; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = 0.1; % high cut-off of the bandpass filter 0.1
params.wsize = 30; % size of the FCD windows
params.overlap = 28; % overlap of the FCD windows
% IT IS TR=2 FOR ENZOS SLEEP DATA
params.TR = 2; % repetition time of the fMRI signal (will be used to simulate fMRI)
params.batch_size = 50000; % batch for
% circular buffer of simulation
% params.seed = 10; % initial condition for the simulation. 
% Heterogenity
params.receptors = av;
% Setting data constants
params.N=90;
params.NSUB=15;
params.TMAX=198;

Isubdiag = find(tril(ones(params.N),-1));




%% 


indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;
    Wdata(:,:,nsub)=TS_W{1,nsub}(:,1:params.TMAX) ; % TS_W tiene registros de distinta longitud
    Wdata(:,:,nsub) = filter_bold(Wdata(:, :,nsub), params.flp, params.fhi, params.TR);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end

for nsub=indexsub
    nsub;
    N3data(:,:,nsub)=TS_N3{1,nsub}(:,1:params.TMAX) ; % TS_W tiene registros de distinta longitud
    N3data(:,:,nsub) = filter_bold(N3data(:, :,nsub), params.flp, params.fhi, params.TR);
    N3FCdata(nsub,:,:)=corrcoef(squeeze(N3data(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end
WFCdata = permute(WFCdata, [2,3,1]);
N3FCdata = permute(N3FCdata, [2,3,1]);



%fcd = compute_fcd(mean(N3data, 3), params.wsize, params.overlap, Isubdiag);
%nwins = size(fcd,1);
%ave_fc = %mean(N3data,3);
ave_fc = mean(N3FCdata,3);

%% Running optimizer
% Bayes optimization options
% The optimizer runs for a very long time, but it saves in disk the output
% of each iteration, so it can be cancelled and resumed then reading the
% files from disk.
opt_time = 3600*7*24; % a week
checkpoint_folder = 'checkpoints/';
%experiment_name = 'N3_FC_filtered'; % Filtering with G=[]0 2.5
experiment_name = 'N3_FC_filt-order6';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_checkpoint_dmf_bayesopt_N',num2str(params.N),'_v1.mat'];
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file};


% Optimizable parameters
G = [0.1 2.9]; 
alpha_slope = 0.75; % fix alpha-> determines the local inhibitiory feedback. Can be optimized as well.
params.receptors = 0;
nm_scale = 0; % Here only optimizing nm
nm_bias = 0;

T = params.TMAX; % seconds to simulate, should be comparable to the empirical recording time (~7-10 minutes of resting state)

% Optimizing heterogenity parameters
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,[],G,alpha_slope,nm_scale,params,bo_opts); % Optimizes FCD

%%
opt_res = load([checkoint_file]);
[best_pars,est_min_ks] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
%%
experiment_name = 'N3_FC_filtered'; % Filtering with G=[]0 2.5
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_checkpoint_dmf_bayesopt_N',num2str(params.N),'_v1.mat'];
opt_res = load([checkoint_file]);
[best_pars,est_min_ks] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
%% Loading optimization results from disk and resuming optimization
% Assuming training is done, and we have a checkpoint file
opt_res = load(checkoint_file);
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;

% Bayes optimization options
checkoint_file2 = [strcat(basefold, checkpoint_folder),strcat(experiment_name, 'finetune'),'_checkpoint_dmf_bayesopt_N',num2str(N),'_v2.mat'];
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
G = [2.3 2.9];
alpha_slope = 0.75; % fix alpha

T = params.TMAX; % seconds
%[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] =  fit_fc_fcd_dmf_only_slope(T,ave_fc,[],G,alpha_slope,params,bo_opts2); % optimizes FC
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_fc_fcd_dmf_only_slope(T,ave_fc,fcd,G,alpha_slope,params,bo_opts2); % optimizes FCD

%% Extracting Optimal Parameters
opt_res2 = load(checkoint_file2);
[best_pars,est_min_ks] = bestPoint(opt_res2.BayesoptResults,'Criterion','min-mean')

%% Optimizing G for FCD
% TODO: CHECK HOW TO CALCULATE FCD. SHOULD BE NWINS X NWIXS X SUBJECTS
Wdata = permute(Wdata, [2,3,1]);
ave_rec = mean(Wdata, 3);
fcd = compute_fcd(ave_rec, wsize, overlap, Isubdiag);


%%

load('DataSleepW_N3.mat')
s1wake = TS_W{1}
FCempWake = corrcoef(s1wake')
s2sleep = TS_N3{1}
FCempN3 = corrcoef(s2sleep')
load(['Results/','DMF_BOLD_FC_fitted_W.mat'])
FcsimWake = corrcoef(BOLD')
clear BOLD
load(['Results/','DMF_BOLD_FC_fitted_N3.mat'])
FcsimN3 = corrcoef(BOLD')
corrplot(FCempWake)

figure
imagesc(FCempWake)
colormap(gca,'parula');
colorbar();
caxis([-1 1])
axis equal 
axis tight

figure
imagesc(FcsimWake)
colormap(gca,'parula');
colorbar();
caxis([-1 1])
axis equal 
axis tight


figure
imagesc(FCempN3)
colormap(gca,'parula');
colorbar();
caxis([-1 1])
axis equal 
axis tight

figure
imagesc(FcsimN3)
colormap(gca,'parula');
colorbar();
caxis([-1 1])
axis equal 
axis tight


figure
imagesc(FCsimJustG)
colormap(gca,'parula');
colorbar();
caxis([-1 1])
axis equal 
axis tight



