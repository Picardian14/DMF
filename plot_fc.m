function stop = plot_fc(results,state)
persistent hs ssimbest fcbest besthist fctrace
stop = false;
switch state
    case 'initial'
        hs = figure;
        besthist = [];
        %fcbest = 0;
        ssimbest = 0;
        %fctrace = [];
    case 'iteration'
        figure(hs)

        fc = results.UserDataTrace{end}{4};   % get nsupp from UserDataTrace property.

        ssim = results.UserDataTrace{end}{3};
  
        %fctrace(:, end+1) = fc; % accumulate nsupp values in a vector.
        if (results.ObjectiveTrace(end) == min(results.ObjectiveTrace)) || (length(results.ObjectiveTrace) == 1) % current is best
            fcbest = fc;
            ssimbest = ssim;
            imagesc(fcbest);
     
            xlabel 'ROI'
            ylabel 'ROI'
            title 'Simulated corrleation'
            drawnow
        end
  
        besthist = [besthist,fcbest];
        %plot(1:length(fctrace),fctrace,'b',1:length(besthist),besthist,'r--')
        
end