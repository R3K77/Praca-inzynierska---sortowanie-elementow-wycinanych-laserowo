function env = exampleHelperPlotBin(binLength, binWidth, binHeight, binCenterPosition, binRotation)
    %exampleHelperPlotBin Adds a bin to the current axes
    
    
    % Move the light so that the side of the bin is illuminated
    lightObj = findobj(gca,'Type','Light');
    for i = 1:length(lightObj)
        lightObj(i).Position = [1,1,1];
    end
    
    env = {};  % No collision boxes added
    end
    