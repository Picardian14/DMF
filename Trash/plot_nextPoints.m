function stop = plot_nextPoints(results, state)
persistent hs list_ofpoints names points
stop = false;
switch state
    case 'initial'
        points = [];
        hs = figure;             
        names = results.NextPoint.Variables;
        list_ofpoints = arrayfun(@(x) ([]), names, 'UniformOutput', false);
    case 'iteration'
        figure(hs)
        nextPoint = results.NextPoint{1,1};
        points = [points, nextPoint];            
        plot(1:length(points), points , 'r.', 'MarkerSize',10);
        xlabel('Iteration number')
        ylabel('Next Points')
        title('Trace of chosen points')
        drawnow
        %{
        tiledlayout(1,length(names))
        for idx=1:length(results.NextPoint.Variables)            
            nexttile;
            nextPoint = results.NextPoint(1,idx);
            list_ofpoints(idx) = [list_ofpoints(idx), nextPoint];            
            plot(1:length(list_ofpoints(idx) ), list_ofpoints(idx) , 'r.', 'MarkerSize',10);
            xlabel('Iteration number')
            ylabel('Next Points')
            title('Trace of chosen points')
            drawnow
        end
        %}
end
end
