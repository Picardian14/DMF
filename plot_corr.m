function stop = plot_corr(results,state)
persistent hs ssimbest  besthist fctrace
stop = false;
switch state
    case 'initial'
        hs = figure;
        besthist = [];
        ssimbest = 0;
        fctrace = [];
    case 'iteration'
        figure(hs)
        ssim = results.UserDataTrace{end}{4}('corr');
        fctrace(end+1) = ssim; % accumulate nsupp values in a vector.
        if (results.ObjectiveTrace(end) == min(results.ObjectiveTrace)) || (length(results.ObjectiveTrace) == 1) % current is best 
            ssimbest = ssim;
        end
        besthist = [besthist,ssimbest];
        plot(1:length(fctrace),fctrace,'b',1:length(besthist),besthist,'r--')
        xlabel 'Iteration number'
        ylabel 'Corr'
        title 'Corr at iteratioms'
        legend('Current iteration','Best objective','Location','best')
        drawnow
    case 'done'
        savefig(hs, 'Figuras/corr_plot.fig')        
end