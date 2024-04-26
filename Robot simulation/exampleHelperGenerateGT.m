function [partGT,env] = exampleHelperGenerateGT(binLength, binWidth, binHeight, binCenterPosition, binRotation,numberOfParts,env)

%% Define Part propterties
partLength = 0.145; % 4 inch
partheight = 0.0508; % 2 inches
partWidth = 0.0508; % 2 inches

partGT = zeros(numberOfParts,4);

%% Define Bounds
Xmin = binCenterPosition(1) - binLength / 2 + partWidth / 2;
Xmax = binCenterPosition(1) + binLength / 2 - partWidth / 2;
Ymin = binCenterPosition(2) - binWidth / 2 + partWidth / 2;
Ymax = binCenterPosition(2) + binWidth / 2 - partWidth / 2;

%% Generate Ground Truth
for i=1:numberOfParts
    partCenterX = Xmin + (Xmax-Xmin)*rand;
    partCenterY = Ymin + (Ymax-Ymin)*rand;
    partCenterZ = (binCenterPosition(3) - (binHeight/2) + (partheight/2) + 0.002)+0.05;
    partRotation = 360*rand;
    partGT(i,:) = [partCenterX partCenterY partCenterZ partRotation];
end

while 1
    %% Check for collision
    [ommitedParts,env] = omitOverlappingPart(numberOfParts,env,partGT,partLength,partWidth,partheight,binHeight, binCenterPosition,Xmin,Xmax,Ymin,Ymax);

    %% Regenerate Ground Truth for remaining parts
    if ~isempty(ommitedParts)
        partGT = regenerateGT(ommitedParts,partGT,Xmin,Xmax,Ymin,Ymax);
    else
        break;
    end
end

    function partGT = regenerateGT(ommitedParts,partGT,Xmin,Xmax,Ymin,Ymax)
        for k=1:length(ommitedParts)
            partCenterX = Xmin + (Xmax-Xmin)*rand;
            partCenterY = Ymin + (Ymax-Ymin)*rand;
            partCenterZ = (binCenterPosition(3) - (binHeight/2) + (partheight/2) + 0.002)+0.05;
            partRotation = 360*rand;
            partGT(ommitedParts(k),:) = [partCenterX partCenterY partCenterZ partRotation];
        end
    end

end

function [ommitedParts,env] = omitOverlappingPart(numberOfParts,env,partGT,partLength,partWidth,partheight,binHeight, binCenterPosition,Xmin,Xmax,Ymin,Ymax)
ommitedParts = [];
for i=1:numberOfParts
    B(i) = collisionBox(partLength,partWidth,partheight);
    B(i).Pose(1:3,4) = [partGT(i,1) partGT(i,2) (binCenterPosition(3) - (binHeight/2) + (partheight/2) + 0.002)]';
    B(i).Pose(1:3,1:3) = eul2rotm([deg2rad(partGT(i,4)+90) 0 0]);

    collision = false;

    % Check Collision with bin walls
    Xdist = min(abs(partGT(i,1)-Xmin),abs(partGT(i,1)-Xmax));
    Ydist = min(abs(partGT(i,2)-Ymin),abs(partGT(i,2)-Ymax));
    if Xdist < (partLength/2)|| Ydist < (partLength/2)
        for j = 1:5
            collisionStatus = checkCollision(env{j},B(i));
            if collisionStatus
                collision = true;
                if isempty(ommitedParts)
                    ommitedParts = i;
                else
                    ommitedParts(end+1) = i;
                end
                break;
            end
        end
    end

    % Check collision with ohter parts
    for j=1:i
        if i==j
            continue;
        else
            if norm(partGT(i,1:2) - partGT(j,1:2)) < norm([partLength partWidth])
                collisionStatus = checkCollision(B(j),B(i));
                if collisionStatus
                    collision = true;
                    if isempty(ommitedParts)
                        ommitedParts = i;
                    else
                        ommitedParts(end+1) = i;
                    end
                    break;
                end
            end
        end
    end
end

if isempty(ommitedParts)
    for i = 1 : numberOfParts
        env(end+1) = {B(i)};
    end
end


end

