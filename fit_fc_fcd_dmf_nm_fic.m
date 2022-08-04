function [opt_fc_error,opt_fcd_ks,opt_pars,results] = fit_fc_fcd_dmf_nm_fic(T,emp_fc,emp_fcd,G,slope,nm,dmf_pars,opts)
%
% Function to find optimal DMF parameters to fit either the FC or the FCD
% of bold signals. Uses dmf_pars.receptors as input to neuromodulate the
% FIC
% T = seconds to simulate
% tr = scalar, TR of simulated BOLD signals
% emp_fc = N x N empirical FC
% emp_fcd = N x N empirical FCD. If empty, only works with FCD.
% G = 2 x 1 with upper and lower bounds on G or empty
% slope = 2 x 1 with upper and lower bounds on FIC slope or empty
% dmf_pars = structure generarted by dyn_fic_DefaultParams with DMF
% parameters
% opts = options for the fit and optimization.

% Setting DMF parameters
N = size(dmf_pars.C,1);
stren = sum(dmf_pars.C)./2;
isubfc = find(tril(ones(N),-1));
nsteps = T.*(1000); % number of DMF timepoints


opt_vars = [];
if length(G(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize, use fix value provided in dmf_pars
    gvals = optimizableVariable('G',[G(1) G(2)]);
    opt_vars = [gvals];
end

if length(slope(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    alphavals = optimizableVariable('alpha',[slope(1) slope(2)]);
    opt_vars = [opt_vars alphavals];
end

if length(nm(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    nmvals = optimizableVariable('nm',[nm(1) nm(2)]);
    opt_vars = [opt_vars nmvals];
end

% BAYES OPTIMIZATION
if isempty(emp_fcd) % only optimizes FC
    results = bayesopt(@aux_dmf_fit_fc,opt_vars,opts{:});
    opt_fc_error.results = results.MinEstimatedObjective;
    opt_fcd_ks = [];
    opt_pars =results.XAtMinEstimatedObjective;
    
else % OPTIMIZES FCD and returns FC GOF
    results = bayesopt(@aux_dmf_fit_fcd,opt_vars,opts{:});
    opt_fcd_ks = results.MinEstimatedObjective;
    opt_pars =results.XAtMinEstimatedObjective;
    [~,min_id] = min(results.EstimatedObjectiveMinimumTrace);
    opt_fc_error = results.UserDataTrace{min_id};
end


    function [fc_error] = aux_dmf_fit_fc(g_a_nm)
        % Unpacking parameters
        thispars = unpack_parameters(g_a_nm);
        
        % Simulating
        bold = DMF(thispars, nsteps,'bold'); % runs simulation
        bold = bold(:,dmf_pars.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        if isempty(bold)
            fc_error = nan;
            return
        end
        % Filtering and computing FC
        filt_bold = filter_bold(bold',dmf_pars.flp,dmf_pars.fhi,dmf_pars.TR);
        sim_fc = corrcoef(filt_bold);
        
        % Computing error: 1-Corrrelation between FC's
        fc_error= 1-corr2(sim_fc(isubfc),emp_fc(isubfc));
        
    end

    function [fcd_ks,const,fc_error] = aux_dmf_fit_fcd(g_a_nm)
        const = [];
        
        thispars = unpack_parameters(g_a_nm);
        
        % Simulating
        bold = DMF(thispars, nsteps, 'bold'); % Estaba esto, pongo la actual: dyn_fic_DMF(thispars, nsteps,'bold');  runs simulation
        bold = bold(:,dmf_pars.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        % Filtering and computing FC
        filt_bold = filter_bold(bold',dmf_pars.flp,dmf_pars.fhi,dmf_pars.TR);
        sim_fc = corrcoef(filt_bold);
        % Computing FC error: 1-Corrrelation between FC's
        fc_error= 1-corr2(sim_fc(isubfc),emp_fc(isubfc));
        % FCD
        sim_fcd = compute_fcd(filt_bold,dmf_pars.wsize,dmf_pars.overlap,isubfc);
        sim_fcd(isnan(sim_fcd))=0;
        sim_fcd = corrcoef(sim_fcd);
        if isempty(sim_fcd)
            fcd_ks = nan;
            fc_error = nan;
            return
        end
        [~,~,fcd_ks] = kstest2(sim_fcd(:),emp_fcd(:));
    end

    function thispars = unpack_parameters(g_a_nm)
        % Unpacking parameters
        thispars = dmf_pars;
        if ismember('G',g_a_nm.Properties.VariableNames)
            thispars.G = g_a_nm.G;
%             thispars.J = 1.5*thispars.G*stren' + 1; % uses default slope value
        end
        if ismember('alpha',g_a_nm.Properties.VariableNames)            
            thispars.J = g_a_nm.alpha*thispars.G*stren' + 1; % updates FIC
        end
        if ismember('nm',g_a_nm.Properties.VariableNames)
            fic_nm = thispars.receptors.*g_a_nm.nm; % Could add bias
            thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC
        end
        
    
    end
end