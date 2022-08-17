clear all;
close all;
basefold = './data/';
fcd_file = 'DataSleepW_N3'; % file with fc, fcd, fMRI and filter parameters for all subjects
load('data/heterogeneitys/myelin_aal.mat') % Heterogenity
load([basefold,fcd_file,'.mat'])
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
params.N=90;
params.NSUB=15;
params.TMAX=198;
Isubdiag = find(tril(ones(params.N),-1));

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

% Optimizable parameters
G = [1 2.9]; 
alpha_slope = [0.5 1]; % fix alpha-> determines the local inhibitiory feedback. Can be optimized as well.
params.receptors = 0;
nm_scale = 0; % Here only optimizing nm
nm_bias = 0;
T = params.TMAX; % seconds to simulate, should be comparable to the empirical recording time (~7-10 minutes of resting state)
% Optimizing heterogenity parameters


checkpoint_folder = 'checkpoints/';
experiment_name = 'W_MSE_Galpha';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_v1.mat'];
%opt_time = 1800; % half hour
opt_time = 60; % half hour
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_ssim,@plot_fc, @plotObjectiveModel,@plotMinObjective}};

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_MSE(T,ave_fc,[],G,alpha_slope,nm_scale, nm_bias,params,bo_opts); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_mse,est_min_ks_mse] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
experiment_name = 'W_SSIM_Galpha';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_v1.mat'];
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_ssim,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_SSIM(T,ave_fc,[],G,alpha_slope,nm_scale, nm_bias,params,bo_opts); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')

experiment_name = 'W_CORR_Galpha';
checkoint_file = [strcat(basefold, checkpoint_folder),experiment_name,'_v1.mat'];
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_ssim,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out] = fit_CORR(T,ave_fc,[],G,alpha_slope,nm_scale, nm_bias,params,bo_opts); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_corr,est_min_ks_corr] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')


params = DefaultParams();
params.receptors = av/max(av);
stren = sum(params.C);
thispars = params;

thispars.G = 1.81;
thispars.alpha = 0.69;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it
params.nm = -1.4097;
params.nm_bias = -0.6562;
%fic_nm = (thispars.receptors.*params.nm)+params.nm_bias; % Could add bias
%thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC

% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000;
for nsub=1:15
    nsub
    BOLDNM = DMF(thispars, nb_steps);
    
    % Minimal "post-processing": band-pass filter and remove the starting and
    % trailing ends of the simulation to avoid transient and filtering artefacts
    BOLDNM = filter_bold(BOLDNM', params.flp, params.fhi, params.TR);
    BOLDNM = BOLDNM';
    %[B, A] = butter(2, [0.01, 0.1]*2*params.TR);
    %BOLD = filter(B, A, BOLD')';
    
    trans = 5;
    BOLDNM = BOLDNM(:, 1+trans:end-trans);
    simulations(:, :, nsub) = BOLDNM(:, 1:198);
    simulationsFC(:, :, nsub) = corrcoef(squeeze(simulations(:, :, nsub))');
end
