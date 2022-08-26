function [ params ] = DefaultParams(varargin)
%%DEFAULTPARAMS Default parameter settings for DMF simulation
%
%   P = DEFAULTPARAMS() yields a struct with default values for all necessary
%   parameters for the DMF model. The default structural connectivity is the
%   DTI fiber consensus matrix obtained from the HCP dataset, using a
%   Schaeffer-100 parcellation.
%
%   P = DEFAULTPARAMS('key', 'value', ...) adds (or replaces) field 'key' in
%   P with given value.
%
% Pedro Mediano, Feb 2021

params = [];

% Connectivity matrix
if any(strcmp(varargin, 'C'))
  C = [];

else
  try
    p = strrep(mfilename('fullpath'), 'DefaultParams', '');
    basefold = 'data/';
    fcd_file = 'DataSleepW_N3'; % file with fc, fcd, fMRI and filter parameters for all subjects
    load([basefold,fcd_file,'.mat']);
    C=SC/max(max(SC))*0.2;
  catch
    error('No connectivity matrix provided, and default matrix not found.');
  end

end


% DMF parameters

params.burnout = 10; % seconds to remove after initial transient of simulation
params.flp = 0.04; % low cut-off of the bandpass filter 0.01 for aal wake
params.fhi = 0.07; % high cut-off of the bandpass filter 0.1
params.wsize = 30; % size of the FCD windows
params.overlap = 28; % overlap of the FCD windows
% IT IS TR=2 FOR ENZOS SLEEP DATA
params.TR = 2; % repetition time of the fMRI signal (will be used to simulate fMRI)
params.batch_size = 50000; % batch for
params.C         = C;
params.receptors = 0;
params.dt        = 0.1;     % ms
params.taon      = 100;     % NMDA tau ms
params.taog      = 10;      % GABA tau ms
params.gamma     = 0.641;   % Kinetic Parameter of Excitation
params.sigma     = 0.01;    % Noise SD nA
params.JN        = 0.15;    % excitatory synaptic coupling nA
params.I0        = 0.382;   % effective external input nA
params.Jexte     = 1.;      % external->E coupling
params.Jexti     = 0.7;     % external->I coupling
params.w         = 1.4;     % local excitatory recurrence
params.de        = 0.16;    % excitatory non linear shape parameter
params.Ie        = 125/310; % excitatory threshold for nonlineariy
params.g_e       = 310.;    % excitatory conductance
params.di        = 0.087;   % inhibitory non linear shape parameter
params.Ii        = 177/615; % inhibitory threshold for nonlineariy
params.g_i       = 615.;    % inhibitory conductance
params.wgaine    = 0;       % neuromodulatory gain
params.wgaini    = 0;       % neuromodulatory gain

params.G         = 1.7156;       % Global Coupling Parameter
params.alpha = 0.4189;
params.nm = 1; % FIC scaling
params.nm_bias = 0; % FIC scaling
% Data parameters

% Balloon-Windkessel parameters (from firing rates to BOLD signal)
params.TR  = 2;     % number of seconds to sample bold signal
params.dtt = 0.001; % BW integration step, in seconds
params.TMAX=198;
% Parallel computation parameters
params.batch_size = 5000;

% Add/replace remaining parameters
for i=1:2:length(varargin)
  params.(varargin{i}) = varargin{i+1};
end

% If feedback inhibitory control not provided, use heuristic
if ~any(strcmp(varargin, 'J'))
  params.J = 0.75*params.G*sum(params.C, 1)' + 1;
end

end

