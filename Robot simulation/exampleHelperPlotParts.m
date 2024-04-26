function exampleHelperPlotParts(binLength, binWidth, binHeight, binCenterPosition, binRotation, partGT)
%exampleHelperPlotBin Adds objects to the current axes

%   Copyright 2021 The MathWorks, Inc.

% Load handy box
FV = stlread(strcat('meshes',filesep,'box_part.STL'))

partheight = 0.0508; % 2 inch
partLength = 0.1016; % 4 inch
rotationZ = @(t) [cosd(t) -sind(t) 0; sind(t) cosd(t) 0; 0 0 1];

    for i=1:size(partGT,1)

    p(i) = patch(gca,'Faces',FV.ConnectivityList,'Vertices',((FV.Points*rotationZ(-partGT(i,4)))...
        + [partGT(i,1), partGT(i,2), (binCenterPosition(3)- (binHeight/2))]),'FaceColor',[0.8 0.8 1.0],'Tag','part');

    end
end