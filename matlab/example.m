%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Example usage of DMF fMRI simulator.
%
% Pedro Mediano, Apr 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/heterogeneitys/myelin_aal.mat') % Heterogenity
% Fetch default parameters
%load('../data/heterogeneitys/myelin_aal.mat') % Heterogenity
params = DefaultParams();
params.receptors = av/max(av);
stren = sum(params.C);
thispars = params;

thispars.G = 1.81;
thispars.alpha = 0.69;
thispars.J = thispars.alpha*thispars.G*stren' + 1; % updates it
params.nm = -1.4097;
params.nm_bias = -0.6562;
fic_nm = (thispars.receptors.*params.nm)+params.nm_bias; % Could add bias
thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC

% Run simulation for a given nb of steps (milliseconds)
nb_steps = 500000;
for nsub=1:15
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
end



