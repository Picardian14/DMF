%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example usage of DMF fMRI simulator.
%
% Pedro Mediano, Apr 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fetch default parameters
%load('../data/heterogeneitys/myelin_aal.mat') % Heterogenity
params = DefaultParams();
params.receptors = 0;
stren = sum(params.C);
thispars = params;

thispars.G = params.G;
thispars.J = 0.75*thispars.G*stren' + 1; % updates it
fic_nm = thispars.receptors.*params.nm; % Could add bias
%thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC

% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000;
BOLD = DMF(thispars, nb_steps);

% Minimal "post-processing": band-pass filter and remove the starting and
% trailing ends of the simulation to avoid transient and filtering artefacts
BOLD = filter_bold(BOLD', params.flp, params.fhi, params.TR);
BOLD = BOLD'
%[B, A] = butter(2, [0.01, 0.1]*2*params.TR);
%BOLD = filter(B, A, BOLD')';

trans = 5;
BOLD = BOLD(:, 1+trans:end-trans);


