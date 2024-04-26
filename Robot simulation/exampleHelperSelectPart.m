function [RefPose,partID,goalZoffset] = exampleHelperSelectPart(partGT,binCenterPosition)

% Select the part which is closest to the center
% Find euclidean distance from bin center
dist = zeros(1,size(partGT,1));
for i=1:size(partGT,1)
    dist(i) = norm(partGT(i,1:2) - binCenterPosition(1:2));
end

[~,I] = min(dist);

goalZoffset = 0.01;

RefPose = [partGT(I,1) partGT(I,2) partGT(I,3)+0.02];
partID = I;

