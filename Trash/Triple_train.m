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

%ave_fc = mean(N3FCdataF,3);
ave_fc = mean(WFCdataF,3);


opt_time_1 = 1200; % 20 min
opt_time_2 = 600; % 10 min


experiment_name = "W_tripletrain";
if ~exist(fullfile("Figuras",experiment_name))
    mkdir(fullfile("Figuras",experiment_name))
end
if ~exist(fullfile("Results",experiment_name))
    mkdir(fullfile("Results",experiment_name))
end
   
if ~exist(fullfile("data/checkpoints",experiment_name))
    mkdir(fullfile("data/checkpoints",experiment_name))
end

%
% SSIM
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
checkoint_file = [fullfile(checkpoint_folder,filename+"_v1.mat")];

%opt_ssim = load('W_SSIM_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat');
%best_pars_ssim = bestPoint(opt_ssim.BayesoptResults,'Criterion','min-mean');
% Optimizable parameters
G = [1 2.9];%best_pars_ssim.G; 
alpha = [0.1 1];%best_pars_ssim.alpha; 

gain_exc = [-3 3]; % Here only optimizing gain
gain_inh = 0;%[-2 2];


bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_1,...
        'OutputFcn',@saveToFile,...
        'SaveFileName',checkoint_file,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc, @plotObjectiveModel,@plotMinObjective}};
[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = gain_fit_with_metrics(params.TMAX,ave_fc,[],G,alpha,gain_exc, gain_inh,params,bo_opts, 'ssim'); % Optimizes FCD
opt_res = load([checkoint_file]);
[best_pars_ssim,est_min_ks_ssim] = bestPoint(opt_res.BayesoptResults,'Criterion','min-mean')
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name))
%movefile("Results/FCtrace.mat", fullfile("Results", experiment_name, sub_experiment_name, "FCtrace_base.mat"))
close all;
% Finetuning
iniX = opt_res.BayesoptResults.XTrace;
iniObj = opt_res.BayesoptResults.ObjectiveTrace;
checkoint_file2 = [fullfile(checkpoint_folder,filename+"_v2.mat")];
bo_opts2 = {'InitialX',iniX,'InitialObjective',iniObj,...
    'IsObjectiveDeterministic',false,'UseParallel',true,...
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',opt_time_2   ,...
        'OutputFcn',@saveToFile,...
        'PlotFcn', {@plot_mse,@plot_ssim,@plot_corr,@plot_fc,@plotMinObjective},...
        'SaveFileName',checkoint_file2};
G = [max(1,best_pars_ssim.G-0.5) min(best_pars_ssim.G+0.5, 2.9)];
alpha = [max(0.1,best_pars_ssim.alpha-0.2) min(best_pars_ssim.alpha+0.2, 1)];
gain_exc = [max(-3,best_pars_ssim.gain_exc-0.5) min(best_pars_ssim.gain_exc+0.5, 3)];
gain_inh = 0;%[max(-2,best_pars_ssim.gain_inh-0.5) min(best_pars_ssim.gain_inh+0.5, 2)];

[opt_fc_error,opt_fcd_ks,opt_pars,bayesopt_out_ssim] = gain_fit_with_metrics(params.TMAX,ave_fc,[],G,alpha,gain_exc, gain_inh,params,bo_opts2, 'ssim'); % Optimizes FCD
best_pars_ssim = bestPoint(bayesopt_out_ssim, 'Criterion', 'min-mean')
save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "best_pars_ssim", "bayesopt_out_ssim")
if ~exist(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
    mkdir(fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
end
movefile("Figuras/*plot.fig",fullfile("Figuras",experiment_name, sub_experiment_name, "Finetune"))
close all;


% Simulations


params = DefaultParams();
params.receptors = av/max(av);
params.receptors(find(params.receptors==0))=mean(params.receptors);
stren = sum(params.C);
thispars = params;

% SSIM SIM
sub_experiment_name = "SSIM";
%opt_ssim = load('W_SSIM_Galphafinetune_checkpoint_dmf_bayesopt_N90_v2.mat');
%best_pars_ssim_galpha = bestPoint(opt_ssim.BayesoptResults,'Criterion','min-mean');
thispars.G = best_pars_ssim.G;
thispars.alpha = best_pars_ssim.alpha;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it
thispars.gaine = best_pars_ssim.gain_exc;
%thispars.wgaine = best_pars_ssim.gain_exc;
%thispars.wgaini = 0.2838;
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

save_name = fullfile("Results", experiment_name, sub_experiment_name, sub_experiment_name+".mat");
save(save_name, "simulationsFC")
h = figure();
imagesc(squeeze(mean(simulationsFC ,3)));
savefig(h, fullfile("Figuras",experiment_name, sub_experiment_name, sub_experiment_name+"_SIM.fig"))
disp(1-ssim(ave_fc, mean(simulationsFC ,3)))
