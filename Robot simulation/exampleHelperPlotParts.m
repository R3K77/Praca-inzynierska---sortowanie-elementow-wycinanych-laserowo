function parts = exampleHelperPlotParts(binLength, binWidth, binHeight, binCenterPosition, binRotation, partGT)
%exampleHelperPlotBin Adds objects to the current axes

%   Copyright 2021 The MathWorks, Inc.
parts = [];
% Load handy box
% Load STL files starting with "output"
stlFiles = dir('meshes/output_*.stl');


for i = 1:length(stlFiles)
    FV = stlread(fullfile('meshes', stlFiles(i).name));
    
    rotationZ = @(t) [cosd(t) -sind(t) 0; sind(t) cosd(t) 0; 0 0 1];

    for i=1:size(partGT,1)
    
        p = patch(gca, 'Faces', FV.ConnectivityList, 'Vertices', ((FV.Points * rotationZ(-partGT(4)))...
                + [partGT(1), partGT(2), 0]), 'FaceColor', [0.8 0.8 1.0], 'Tag', 'part');
                parts = [parts, p];
    end
    
end