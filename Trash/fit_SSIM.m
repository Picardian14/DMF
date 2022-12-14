function [opt_fc_error,opt_fcd_ks,opt_pars,results  ] = fit_SSIM(T,emp_fc,emp_fcd,G,slope, nm, nm_bias,dmf_pars,opts)
%
% Function to find optimal DMF parameters to fit either the FC or the FCD
% of bold signals.
% T = seconds to simulate
% tr = scalar, TR of simulated BOLD signals
% emp_fc = N x N empirical FC
% emp_fcd = N x N empirical FCD. If empty, only works with FC.
% G = 2 x 1 with upper and lower bounds on G or empty
% slope = 2 x 1 with upper and lower bounds on FIC slope or empty
% dmf_pars = structure generarted by dyn_fic_DefaultParams with DMF
% parameters
% opts = options for the fit and optimization.

% Setting DMF parameters
N = size(dmf_pars.C,1);
stren = sum(dmf_pars.C);
isubfc = find(tril(ones(N),-1));
nsteps = T.*(1000); % number of DMF timepoints
gamma_ent_fun = @(a) a(1) + log(a(2)) + log(gamma(a(1))) + (1-a(1))*psi(a(1));


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

if length(nm_bias(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    nmbiasvals = optimizableVariable('nm_bias',[nm_bias(1) nm_bias(2)]);
    opt_vars = [opt_vars nmbiasvals];
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

    % FITS FC
    function [fc_error,const,outdata] = aux_dmf_fit_fc(g_alpha) 
        const = [];
        % Unpacking parameters
        thispars = dmf_pars;
        thispars.J = 0.75*thispars.G*stren' + 1;
        if ismember('G',g_alpha.Properties.VariableNames)
            thispars.G = g_alpha.G;
            thispars.J = 0.75*thispars.G*stren' + 1; % updates it
        end
        if ismember('alpha',g_alpha.Properties.VariableNames)
            thispars.J = g_alpha.alpha*thispars.G*stren' + 1; % updates it
        end
        if ismember('nm',g_alpha.Properties.VariableNames)
            fic_nm = thispars.receptors.*g_alpha.nm; % Could add bias
            thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC
        end
        if ismember('nm_bias',g_alpha.Properties.VariableNames)
            fic_nm = thispars.receptors.*g_alpha.nm + g_alpha.nm_bias; % Could add bias
            thispars.J = thispars.J + (thispars.J).*fic_nm; % modulates FIC
        end
        
        % Simulating
        [rates,bold] = DMF(thispars, nsteps,'both'); % runs simulation
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
        
        % Computing rates
        rates = rates(:,(dmf_pars.burnout*1000):end);
        reg_fr = mean(rates,2);
        reg_ent = zeros(N,1);
        for n=1:N
            gamma_pars = gamfit(rates(n,:));
            reg_ent(n) = gamma_ent_fun(gamma_pars);
        end
       
        % Computing FC error: Mean Squared differences between vectorized FCs
        
      
        mse_error= mean((sim_fc(isubfc)-emp_fc(isubfc)).^2); % MSE FC
        fc_error = 1-ssim(emp_fc,sim_fc);
        corr_error = 1-corr2(emp_fc, sim_fc);
        outdata = {reg_fr,reg_ent, corr_error, sim_fc, mse_error};
        
    end

    % FITS FCD
    function [fcd_ks,const,outdata] = aux_dmf_fit_fcd(g_alpha)
        const = [];
        % Unpacking parameters
        thispars = dmf_pars;        
        if ismember('G',g_alpha.Properties.VariableNames)
            thispars.G = g_alpha.G;
            thispars.J = 0.75*thispars.G*stren' + 1; % uses default slope value
        end
        if ismember('alpha',g_alpha.Properties.VariableNames)
            thispars.J = g_alpha.alpha*thispars.G*stren' + 1; % updates it
        end
        
        % Simulating
%         bold = dyn_fic_DMF(thispars, nsteps,'bold'); % runs simulation
        [rates,bold] = DMF(thispars, nsteps,'both'); % runs simulation  with rates
        bold = bold(:,dmf_pars.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        % Computing rates
        rates = rates(:,(dmf_pars.burnout*1000):end);
        reg_fr = mean(rates,2);
        reg_ent = zeros(N,1);
        for n=1:N
            gamma_pars = gamfit(rates(n,:));
            reg_ent(n) = gamma_ent_fun(gamma_pars);
        end
        % Filtering and computing FC
        filt_bold = filter_bold(bold',dmf_pars.flp,dmf_pars.fhi,dmf_pars.TR);
        sim_fc = corrcoef(filt_bold);
        % Computing FC error: Mean Squared differences between vectorized FCs
        fc_error= mean((sim_fc(isubfc)-emp_fc(isubfc)).^2); % MSE FC
        outdata = {reg_fr,reg_ent,fc_error};
        % FCD
        sim_fcd = compute_fcd(filt_bold,dmf_pars.wsize,dmf_pars.overlap,isubfc);
        sim_fcd(isnan(sim_fcd))=0;
        sim_fcd = corrcoef(sim_fcd);
        if isempty(sim_fcd)
            fcd_ks = nan;
            outdata = nan;
            return
        end
        [~,~,fcd_ks] = kstest2(sim_fcd(:),emp_fcd(:));
    end
end