function stop = plot_ssim(results,state)
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
        ssim = results.UserDataTrace{end}{3};
        fctrace(end+1) = ssim; % accumulate nsupp values in a vector.
        if (results.ObjectiveTrace(end) == min(results.ObjectiveTrace)) || (length(results.ObjectiveTrace) == 1) % current is best 
            ssimbest = ssim;
        end
        besthist = [besthist,ssimbest];
        plot(1:length(fctrace),fctrace,'b',1:length(besthist),besthist,'r--')
        xlabel 'Iteration number'
        ylabel 'Number of support vectors'
        title 'Number of support vectors at each iteration'
        legend('Current iteration','Best objective','Location','best')
        drawnow
        
end