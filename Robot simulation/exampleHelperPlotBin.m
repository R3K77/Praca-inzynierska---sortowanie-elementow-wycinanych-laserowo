function env = exampleHelperPlotBin(binLength, binWidth, binHeight, binCenterPosition, binRotation)
%exampleHelperPlotBin Adds a bin to the current axes

%   Copyright 2021 The MathWorks, Inc.

% Move the light so that the side of the cup is illuminated
lightObj = findobj(gca,'Type','Light');
for i = 1:length(lightObj)
    lightObj(i).Position = [1,1,1];
end

% Add CollisionBox face 1 (Thickness 0.001 assumed)
b1 = collisionBox(binLength,0.001,binHeight);
b1.Pose(1:3,1:3) = eul2rotm([0 binRotation*pi/180 0]);
b1.Pose(1:3,4) = binCenterPosition+[0 binWidth/2 0];

% Add CollisionBox face 2 (Thickness 0.001 assumed)
b2 = collisionBox(binLength,0.001,binHeight);
b2.Pose(1:3,1:3) = eul2rotm([0 binRotation*pi/180 0]);
b2.Pose(1:3,4) = binCenterPosition+[0 -binWidth/2 0];

% Add CollisionBox face 3 (Thickness 0.001 assumed)
b3 = collisionBox(0.001,binWidth,binHeight);
b3.Pose(1:3,1:3) = eul2rotm([0 binRotation*pi/180 0]);
b3.Pose(1:3,4) = binCenterPosition + (eul2rotm([0 binRotation*pi/180 0])*[binLength/2;0;0])';

% Add CollisionBox face 4 (Thickness 0.001 assumed)
b4 = collisionBox(0.001,binWidth,binHeight);
b4.Pose(1:3,1:3) = eul2rotm([0 binRotation*pi/180 0]);
b4.Pose(1:3,4) = binCenterPosition + (eul2rotm([0 binRotation*pi/180 0])*[-binLength/2;0;0])';

% Add CollisionBox face 5 (Thickness 0.001 assumed)
b5 = collisionBox(binLength,binWidth,0.001);
b5.Pose(1:3,1:3) = eul2rotm([0 binRotation*pi/180 0]);
b5.Pose(1:3,4) = binCenterPosition + (eul2rotm([0 binRotation*pi/180 0])*[0;0;-binHeight/2])';

% Add CollisionBox Place Table (Thickness 0.05 assumed)
table = collisionBox(0.5,0.9,0.05);
table.Pose(1:3,1:3) = eul2rotm([0 0 0]);
table.Pose(1:3,4) = [0 0.9 -0.09];

camera = collisionBox(0.04,0.1,0.04);
camera.Pose(1:3,1:3) = eul2rotm([0 0 0]);
camera.Pose(1:3,4) = [0.54 0 0.61];

show(b1);
show(b2);
show(b3);
show(b4);
show(b5);
[~,patchobj] = show(table);
patchobj.FaceColor = [1 0 0];
show(camera);

env={b1,b2,b3,b4,b5,table,camera};
end
