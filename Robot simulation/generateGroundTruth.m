function [partGT, env] = generateGroundTruth(numberOfParts, env)
    % Initialize partGT with positions from CSV
    partGT = [[ones(numberOfParts, 1)*0.15, zeros(numberOfParts, 1)], zeros(numberOfParts, 1), zeros(numberOfParts, 1)];
    
    % Add a fixed Z-coordinate and rotation
    fixedZ = -0.001;  % Assuming part height of 0.0508
    partGT(:, 3) = fixedZ;  % Fixed Z-coordinate
    partGT(:, 4) = 0;  % Fixed rotation (if needed)
    
    % Load and place parts
    for i = 1:numberOfParts
        filename = sprintf('meshes/output_%d.stl', i - 1);
        FV = stlread(filename);
        % Create a collision box for each part
        partDimensions = max(FV.Points) - min(FV.Points);
        partLength = partDimensions(1);
        partWidth = partDimensions(2);
        partHeight = partDimensions(3);
        B(i) = collisionBox(partLength, partWidth, partHeight);
        B(i).Pose(1:3, 4) = [partGT(i, 1), partGT(i, 2), fixedZ]';
        B(i).Pose(1:3, 1:3) = eul2rotm([deg2rad(partGT(i, 4) + 90), 0, 0]);
    end
    
    % Update the environment with the collision boxes
    for i = 1:numberOfParts
        env(end + 1) = {B(i)};
    end
end
