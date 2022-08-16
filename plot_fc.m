function stop = plot_fc(results,state)
persistent hs ssimbest fcbest besthist fctrace i
stop = false;
switch state
    case 'initial'
        hs = figure;
        besthist = [];
        %fcbest = 0;
        ssimbest = 0;
        fctrace = [];
        i=1;
    case 'iteration'
        figure(hs)

        fc = results.UserDataTrace{end}{4};   % get nsupp from UserDataTrace property.

        ssim = results.UserDataTrace{end}{3};
  
        %fctrace(:, end+1) = fc; % accumulate nsupp values in a vector.
        if (results.ObjectiveTrace(end) == min(results.ObjectiveTrace)) || (length(results.ObjectiveTrace) == 1) % current is best
            fcbest = fc;
            ssimbest = ssim;
            imagesc(fcbest);
            fctrace(end+1)=fc;
            xlabel 'ROI'
            ylabel 'ROI'
            title 'Simulated corrleation'
            drawnow
            disp(hs);
          
            
        end
        
        
        besthist = [besthist,fcbest];
        %plot(1:length(fctrace),fctrace,'b',1:length(besthist),besthist,'r--')
        
end