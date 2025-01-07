close all
clear
clc

clear selectPart

% Load the robot model
ur5eRBT = loadrobot('universalUR5e','DataFormat', 'row', 'Version', 1);

% figure;
% show(ur5eRBT,'Frames','off','PreservePlot',false,'Collisions','off','Visuals','on');


ur5e = addGripper(ur5eRBT);

% figure;
% showdetails(ur5e);
% show(ur5e);

% Home Position
homePosition = deg2rad([-15 -126 113 -80 -91 76]);

partCounter = 0;
element_rotation = [0 0 0 pi 0 0 0];
num = 1;

% Set home position of each joint
ur5e.Bodies{1, 3}.Joint.HomePosition = homePosition(1);
ur5e.Bodies{1, 4}.Joint.HomePosition = homePosition(2);
ur5e.Bodies{1, 5}.Joint.HomePosition = homePosition(3);
ur5e.Bodies{1, 6}.Joint.HomePosition = homePosition(4);
ur5e.Bodies{1, 7}.Joint.HomePosition = homePosition(5);
ur5e.Bodies{1, 8}.Joint.HomePosition = homePosition(6);

% Show robot at home position
f1 = figure;
show(ur5e,homePosition,'Frames','off','PreservePlot',false,'Collisions','off','Visuals','on');
hold on

% Bin dimension
binLength = 0.5; % Along X axis
binWidth = 0.5; % Along Y axis
binHeight = 0.0;
binCenterPosition = [0.65 0.5 0];
binRotation = 0;

% Initialize environment
env = {};
hold on

% Load parts and generate ground truth
data = readmatrix("element_details.csv");
data = sortrows(data, 1);
dataElement = data(:, 2:3);
dataBox = data(:, 5:7);
goalPoints = dataElement/1000;
goalBoxes = dataBox/1000;
numberOfParts = size(goalPoints, 1);

[partGT, env] = generateGroundTruth(numberOfParts, env);

listSTL = [];

% Plot Parts
partHandles = gobjects(numberOfParts, 1);
for i = 1:numberOfParts
    data = readtable('element_details.csv');
    stl_file = data.Nazwa{i};
    filename = strcat('meshes/', stl_file, '.stl');
    FV = stlread(filename);
    partHandles(i) = patch(gca, 'Faces', FV.ConnectivityList, 'Vertices', FV.Points + partGT(i, 1:3), 'FaceColor', [0.8 0.8 1.0], 'Tag', 'part');
end

% Adjust the camera and axis limits
axis([-0.25 1.0 -0.6 0.6 -0.1 0.9]);
view([-90 90]);

drawnow;

% Flag for visualization
showAnimation = true;
showAnimationTraj = false;

% Flag for design
isDesign = true;
placeAngleOffset = 79.0222; % Only applicable when not using IK

% Threshold for pose error
poseThr = 1e-3;

% Radius to switch planner
planRadius1 = 0.18;
planRadius2 = 0.21;

RemainingParts = numberOfParts;

goalFrame = robotics.manip.internal.workspaceGoalRegion.FrameVisual;

% Initialize the list of moved parts
movedParts = [];

simStartTime = tic; % Rozpoczęcie timera

% Inicjalizacja logów
timeLog = [];
xLog = [];
yLog = [];
zLog = [];
thetaLog = [];

for p = 1:numberOfParts
    disp("Starting computation for approach trajectory");

    % Define Goal Region
    goalRegion = workspaceGoalRegion(ur5e.BodyNames{end});

    % Add bounds for the goal region
    goalRegion.Bounds(1, :) = [0 0];  % X Bounds
    goalRegion.Bounds(2, :) = [0 0];  % Y Bounds
    goalRegion.Bounds(3, :) = [0 0];  % Z Bounds
    goalRegion.Bounds(4, :) = [0 0];  % Rotation about the Z-axis
    goalRegion.Bounds(5, :) = [0 0];  % Rotation about the Y-axis
    goalRegion.Bounds(6, :) = [0 0];  % Rotation about the X-axis

    % Select Part to pick using the part selection algorithm
    [RefPose, partID, goalZoffset] = selectPart(partGT, goalPoints);

    % Create goal region based on part position and its reference pose
    goalRegion.ReferencePose = trvec2tform(RefPose);
    goalRegion.ReferencePose(1:3,1:3) = eul2rotm([0 deg2rad(binRotation) 0]);
    goalRegion.EndEffectorOffsetPose = trvec2tform([0 0 goalZoffset]);
    goalRegion.EndEffectorOffsetPose = ...
        goalRegion.EndEffectorOffsetPose*eul2tform([0 pi 0],"ZYX");

    % Visualize the goal region for the approach
    figure(f1);
    hold on
    show(goalRegion);
    drawnow;

    % Create RRT planner using robot rigid body tree and the available
    % collision environment
    planner = manipulatorRRT(ur5e,env);

    % Set planner parameters
    planner.MaxConnectionDistance = 0.1;
    planner.ValidationDistance = 0.1;
    planner.SkippedSelfCollisions = 'adjacent';
    numIter = 50;

    % Plan approach trajectory and compute path
    path = plan(planner,homePosition,goalRegion);

    % Compute short and interpolated path using the computed path
    shortPath = path;
    interpConfigurations = interpolate(planner,shortPath);
    cumulativePath{1} = interpConfigurations;

    % Show animation of the interpolated path
    if showAnimation
        rateObj = rateControl(60);
        for i = 1 : size(interpConfigurations, 1)
            show(ur5e,interpConfigurations(i,:),'PreservePlot',false,'Frames','off','Collisions','off','Visuals','on','FastUpdate',true);
            drawnow
            
            endEffectorName = 'Bellow';
            endEffectorPose = getTransform(ur5e, interpConfigurations(i,:), 'Bellow');
            
            % Ekstrakcja pozycji (x, y, z)
            position = endEffectorPose(1:3, 4);
            x = position(1);
            y = position(2);
            z = position(3);
            
            % Ekstrakcja kąta obrotu wokół osi Z
            R = endEffectorPose(1:3, 1:3);
            theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

            currentTime = toc(simStartTime);

            timeLog = [timeLog; currentTime];
            xLog = [xLog; x];
            yLog = [yLog; y];
            zLog = [zLog; z];
            thetaLog = [thetaLog; theta];

            waitfor(rateObj);
        end
    end
    hold off

    endEffectorApproachTransform = getTransform(ur5e,interpConfigurations(end,:),ur5e.BodyNames{end});
    eulerAtEndEffector = tform2eul(endEffectorApproachTransform);
    
    % Zapisz przesunięcie i orientację elementu względem chwytaka
    elementOffset = [-goalPoints(p, :) 0.025]; % Przesunięcie elementu względem chwytaka 001721
    elementRotation = -deg2rad(partGT(partID,4)) + eulerAtEndEffector(1) + pi; % Orientacja elementu względem chwytaka

    % Add part as a collision box with the robot end-effector tool
    ur5e = attachPart(ur5e, elementRotation, partID, elementOffset, elementOffset);

    showdetails(ur5e);

    % goalPoints(1, :) = [];

    % Remove part from the collision environment (remove part from the bin
    % as it attached to the robot end-effector tool)
    if 7 + partID <= length(env)
        env(7 + partID) = [];
    else
        disp(['Warning: Attempted to delete index ', num2str(7 + partID), ' from env but length(env) is ', num2str(length(env))]);
    end

    %%%%%% Retract trajectory %%%%%%
    disp("Starting computation for retract trajectory")

    % Update the Planner for the modified robot and collision environment
    planner = manipulatorRRT(ur5e,env);

    % Update the planner parameters based on the distance of the object to
    % the center of the bin. Below code set the planner parameter based on
    % the environment complexity to optimize the planning computation
    dist = norm(partGT(partID,1:2) - binCenterPosition(1:2));
    shortFlag = false;
    if dist < planRadius1
        planner.MaxConnectionDistance = 0.05;
        planner.ValidationDistance = planner.MaxConnectionDistance/4;
    elseif dist >  planRadius2
        planner.MaxConnectionDistance = 0.15;
        planner.ValidationDistance = planner.MaxConnectionDistance/2;
        shortFlag = true;
    else
        planner.MaxConnectionDistance = 0.1;
        planner.ValidationDistance = planner.MaxConnectionDistance/3;
        shortFlag = true;
    end
    planner.SkippedSelfCollisions='parent';
    planner.IgnoreSelfCollision = true;

    % Compute the end pose for the retract based on the available parameters
    % using a goal region. If you have computed this once then you can set
    % "isDesign" to false and use the pre-computed configuration to save
    % the computation time.
    if isDesign
        % Define Goal Region
        goalRegion = workspaceGoalRegion(ur5e.BodyNames{end});
        goalRegion.ReferencePose(1:3,1:3) = endEffectorApproachTransform(1:3,1:3);

        goalRegion.Bounds(1, :) = [-0.01 0.01];  % X Bounds
        goalRegion.Bounds(2, :) = [-0.01 0.01];  % Y Bounds
        goalRegion.Bounds(3, :) = [-0.01 0.01];  % Z Bounds
        goalRegion.Bounds(4, :) = [-2*pi 2*pi];  % Rotation about the Z-axis
        goalRegion.Bounds(5, :) = [0 0];  % Rotation about the Y-axis
        goalRegion.Bounds(6, :) = [0 0];  % Rotation about the X-axis

        goalRegion.ReferencePose(1:3,4) = [goalBoxes(partID, 1) goalBoxes(partID, 2) goalBoxes(partID, 3) + 0.1]' + [0.25 -0.85 0.00]';
        goalRegion.EndEffectorOffsetPose = trvec2tform([0 0 0.0]);
        goalRegion.EndEffectorOffsetPose = ...
            goalRegion.EndEffectorOffsetPose*eul2tform([0 0 0],"ZYX");

        % Show goal region
        hold on;
        show(goalRegion);
        drawnow;

        % Compute path for retract based on the given goal region
        path = plan(planner,interpConfigurations(end,:),goalRegion);
    else
        % Pre-computed retract goal configuration
        goalConfiguration = [-0.2240   -1.3443    1.2348   -1.4613   -1.5708    1.3468];

        % Show goal frame
        goalFrame.Pose = getTransform(ur5e,goalConfiguration,ur5e.BodyNames{end});
        hold on;
        show(goalFrame,gca);
        drawnow;

        % Compute path for retract based on the given goal configuration
        path = plan(planner,interpConfigurations(end,:),goalConfigurarion);
    end

    % Compute the short and interpolated path
    if shortFlag
        shortPath = shorten(planner,path,numIter);
        interpConfigurations = interpolate(planner,shortPath);
    else
        shortPath = path;
        interpConfigurations = interpolate(planner,shortPath);
    end
    cumulativePath{2} = interpConfigurations;

    % Delete Goal Region from the figure
    delete(findobj(f1,'type', 'hgtransform'));
    hold on;

    if isDesign
        show(goalRegion);
    else
        show(goalFrame,gca);
    end

    % Delete the part from the figure
    if isvalid(partHandles(partID))
        delete(partHandles(partID));
    end

    % Show animation of the interpolated configuration for retract trajectory
    if showAnimation
        rateObj = rateControl(60);
        for i = 1 : 1 : size(interpConfigurations, 1)
            show(ur5e,interpConfigurations(i,:),'PreservePlot',false,'Frames','off','Collisions','off','Visuals','on','FastUpdate',true);
            
            endEffectorName = 'Bellow';
            endEffectorPose = getTransform(ur5e, interpConfigurations(i,:), 'Bellow');
            
            % Ekstrakcja pozycji (x, y, z)
            position = endEffectorPose(1:3, 4);
            x = position(1);
            y = position(2);
            z = position(3);
            
            % Ekstrakcja kąta obrotu wokół osi Z
            R = endEffectorPose(1:3, 1:3);
            theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

            currentTime = toc(simStartTime);

            timeLog = [timeLog; currentTime];
            xLog = [xLog; x];
            yLog = [yLog; y];
            zLog = [zLog; z];
            thetaLog = [thetaLog; theta];
            
            waitfor(rateObj);
        end
    end

    % Show robot last frame
    show(ur5e,interpConfigurations(end,:),'PreservePlot',false,'Frames','off','Collisions','off','Visuals','on','FastUpdate',true);

    %%%%%% Place trajectory %%%%%%
    % Compute end pose for placing the object
    if isDesign
        % Fixed End-Position so using IK instead of the work space goal region
        targetPoseAngle = [-deg2rad(partGT(partID,4)) + eulerAtEndEffector(1) - pi/2, pi, 0];
        num = num + 1;
    
        aboveTargetPoseXYZ = [goalBoxes(partID, 1), goalBoxes(partID, 2), goalBoxes(partID, 3)] + [0.25, -0.85, 0];
        aboveTargetPoseAngle = [-deg2rad(partGT(partID,4)) + eulerAtEndEffector(1) - pi/2, pi, 0];
        aboveTargetPose = trvec2tform(aboveTargetPoseXYZ) * eul2tform(aboveTargetPoseAngle, "ZXY");
    
        ik = inverseKinematics('RigidBodyTree', ur5e);
        ik.SolverParameters.AllowRandomRestart = false;
        ik.SolverParameters.GradientTolerance = 1e-13;
        ik.SolverParameters.MaxTime = 5;
        weights = [1, 1, 1, 1, 1, 1];
        initialGuess = interpConfigurations(end, :);
        [configAbove, solnInfo] = ik(ur5e.BodyNames{end}, aboveTargetPose, weights, initialGuess);
    
        if solnInfo.PoseErrorNorm > poseThr
            ik.SolverParameters.MaxTime = 10;
            ik.SolverParameters.AllowRandomRestart = true;
            [configAbove, solnInfo] = ik(ur5e.BodyNames{end}, aboveTargetPose, weights, initialGuess);
        end
    
        planner.MaxConnectionDistance = 0.5;
        planner.ValidationDistance = planner.MaxConnectionDistance / 2;
        path = plan(planner, interpConfigurations(end, :), configAbove);
        shortPath = shorten(planner, path, numIter);
        interpAbove = interpolate(planner, shortPath);

        % interpAbove = [interpAbove; interpAbove];
    
        % Use contopptraj to smooth the trajectory
        maxqd = pi/2; % Maximum joint velocity (rad/s)
        maxqdd = deg2rad(100); % Maximum joint acceleration (rad/s^2)
        vellimits = repmat([-maxqd; maxqd], 1, 6);
        accellimits = repmat([-maxqdd; maxqdd], 1, 6);
    
        [qAbove, ~, ~, ~] = contopptraj(interpAbove', vellimits', accellimits', ...
            'NumSamples', size(interpAbove, 1)*10);
    
        % Animate movement to 100mm above the target pose
        if showAnimation
            rateObj = rateControl(60);
            for i = 1:size(qAbove, 2)
                q = qAbove(:, i)';
                show(ur5e, q, 'PreservePlot', false, 'Frames', 'off','Collisions', 'off', 'Visuals', 'on', 'FastUpdate', true);
                
                endEffectorName = 'Bellow';
                endEffectorPose = getTransform(ur5e, q, 'Bellow');
            
                % Ekstrakcja pozycji (x, y, z)
                position = endEffectorPose(1:3, 4);
                x = position(1);
                y = position(2);
                z = position(3);
                
                % Ekstrakcja kąta obrotu wokół osi Z
                R = endEffectorPose(1:3, 1:3);
                theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

                currentTime = toc(simStartTime);

                timeLog = [timeLog; currentTime];
                xLog = [xLog; x];
                yLog = [yLog; y];
                zLog = [zLog; z];
                thetaLog = [thetaLog; theta];

                waitfor(rateObj);
            end
        end
    
        % === 2) Move down to the target pose ===
        targetPoseXYZ = [goalBoxes(partID, 1), goalBoxes(partID, 2), goalBoxes(partID, 3)] + [0.25, -0.85, 0];
        targetPoseAngle = [-deg2rad(partGT(partID,4)) + eulerAtEndEffector(1) - pi/2 + element_rotation(partID), pi, 0];
        targetPose = trvec2tform(targetPoseXYZ) * eul2tform(targetPoseAngle, "ZXY");
        goalFrame.Pose = targetPose;
    
        [configSoln, solnInfo] = ik(ur5e.BodyNames{end}, targetPose, weights, configAbove);
        if solnInfo.PoseErrorNorm > poseThr
            ik.SolverParameters.MaxTime = 10;
            ik.SolverParameters.AllowRandomRestart = true;
            [configSoln, solnInfo] = ik(ur5e.BodyNames{end}, targetPose, weights, configAbove);
        end
    
        path = plan(planner, qAbove(:, end)', configSoln);
        shortPath = shorten(planner, path, numIter);
        interpPlace = interpolate(planner, shortPath);
    
        % Smooth the trajectory from above to target
        if size(interpPlace, 1) < 2
            interpPlace = [interpPlace; interpPlace];
        end
        [qPlace, ~, ~, ~] = contopptraj(interpPlace', vellimits', accellimits', ...
            'NumSamples', size(interpPlace, 1)*10);
    
        % Animate movement to the target pose
        if showAnimation
            rateObj = rateControl(60);
            for i = 1:size(qPlace, 2)
                q = qPlace(:, i)';
                show(ur5e, q, 'PreservePlot', false, 'Frames', 'off', 'Collisions', 'off', 'Visuals', 'on', 'FastUpdate', true);

                
            endEffectorName = 'Bellow';
            endEffectorPose = getTransform(ur5e, q, 'Bellow');
            
            % Ekstrakcja pozycji (x, y, z)
            position = endEffectorPose(1:3, 4);
            x = position(1);
            y = position(2);
            z = position(3);
            
            % Ekstrakcja kąta obrotu wokół osi Z
            R = endEffectorPose(1:3, 1:3);
            theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

            currentTime = toc(simStartTime);

            timeLog = [timeLog; currentTime];
            xLog = [xLog; x];
            yLog = [yLog; y];
            zLog = [zLog; z];
            thetaLog = [thetaLog; theta];     
                
                waitfor(rateObj);
            end
        end

        % Delete Part from the body after placing the object (Modify the robot rigid body tree environment)
        removeBody(ur5e, 'part');
        RemainingParts = RemainingParts - 1;

        % Update the current configuration
        interpConfigurations = [interpConfigurations; interpAbove];

        % Plotowanie elementu w nowej lokalizacji
        hold on;

        data = readtable('element_details.csv');
        filename = data.Nazwa{partID};
        filename = strcat('meshes/', filename, '.stl');

        FV = stlread(filename);

        % Przygotowanie transformacji
        % Transformacja od układu elementu do układu chwytaka
        if element_rotation(partID) == 0
            elementToGripper = trvec2tform(elementOffset);
        else
            % elementToGripper = trvec2tform(elementOffset) * axang2tform([0 0 1 element_rotation(partID)]);
            % elementToGripper = trvec2tform(elementOffset) * axang2tform([0 0 1 elementRotation + element_rotation(partID)]);

        end
        % elementToGripper = trvec2tform(elementOffset) * axang2tform([0 0 1 elementRotation + element_rotation(partID)]);

        % elementToGripper = trvec2tform(elementOffset) * axang2tform([0 0 1 0]);

        % Transformacja od układu chwytaka do układu globalnego (targetPose)
        % Skumulowana transformacja od układu elementu do układu globalnego
        totalTransform = aboveTargetPose * elementToGripper;


        % Zastosowanie skumulowanej transformacji do wierzchołków
        points = [FV.Points, ones(size(FV.Points, 1), 1)]'; % Dodanie jedynek dla współrzędnych homogenicznych

        if partID == 4
            points(1, :) = -points(1, :) + 0.232075;
            points(2, :) = -points(2, :) + 0.336077;
        end
        transformedPoints = (totalTransform * points)'; % Przemnożenie przez macierz transformacji
        transformedPoints = transformedPoints(:, 1:3); % Wyodrębnienie współrzędnych x, y, z
                
        
        fprintf('Element %d placed at position: %f %f %f\n', partID, totalTransform(1, 4), totalTransform(2, 4), totalTransform(3, 4));
        % Rysowanie elementu
        partHandles(partID) = patch(gca, 'Faces', FV.ConnectivityList, 'Vertices', transformedPoints, ...
                                    'FaceColor', [0.8 0.8 1.0], 'Tag', 'part');
    
        % === 3) Retract to 100mm above the placed part ===
        % Define retractPose 100mm above the target pose
        retractPoseXYZ = [goalBoxes(partID, 1), goalBoxes(partID, 2), goalBoxes(partID, 3) + 0.1] + [0.25, -0.85, 0];
        retractPoseAngle = targetPoseAngle; % Maintain the same orientation
        retractPose = trvec2tform(retractPoseXYZ) * eul2tform(retractPoseAngle, "ZXY");
    
        [configRetract, solnInfo] = ik(ur5e.BodyNames{end}, retractPose, weights, configSoln);
    
        if solnInfo.PoseErrorNorm > poseThr
            ik.SolverParameters.MaxTime = 10;
            ik.SolverParameters.AllowRandomRestart = true;
            [configRetract, solnInfo] = ik(ur5e.BodyNames{end}, retractPose, weights, configSoln);
        end
    
        path = plan(planner, configSoln, configRetract);
        shortPath = shorten(planner, path, numIter);
        interpRetract = interpolate(planner, shortPath);

        % interpRetract = [interpRetract; interpRetract];
    
        % Smooth the retract trajectory
        if size(interpRetract, 1) < 2
            interpRetract = [interpRetract; interpRetract];
        end
        [qRetract, ~, ~, ~] = contopptraj(interpRetract', vellimits', accellimits', ...
            'NumSamples', size(interpRetract, 1)*10);
    
        % Animate movement retracting to above the placed part
        if showAnimation
            rateObj = rateControl(60);
            for i = 1:size(qRetract, 2)
                q = qRetract(:, i)';
                show(ur5e, q, 'PreservePlot', false, 'Frames', 'off', ...
                     'Collisions', 'off', 'Visuals', 'on', 'FastUpdate', true);
                
                     endEffectorName = 'Bellow';
                     endEffectorPose = getTransform(ur5e, q, 'Bellow');
                     
                     % Ekstrakcja pozycji (x, y, z)
                     position = endEffectorPose(1:3, 4);
                     x = position(1);
                     y = position(2);
                     z = position(3);
                     
                     % Ekstrakcja kąta obrotu wokół osi Z
                     R = endEffectorPose(1:3, 1:3);
                     theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z
         
                     currentTime = toc(simStartTime);
         
                     timeLog = [timeLog; currentTime];
                     xLog = [xLog; x];
                     yLog = [yLog; y];
                     zLog = [zLog; z];
                     thetaLog = [thetaLog; theta];
                     waitfor(rateObj);
            end
        end
        
    else
        % Fixed joint configuration based on previous analysis
        configSoln = [1.3792 -1.0821 1.2291 -1.7178 -1.5708 wrapToPi(deg2rad(partGT(partID, 4) + placeAngleOffset))];
        goalFrame.Pose = getTransform(ur5e, configSoln, ur5e.BodyNames{end});
    end

    % Parameters for the planner for the place trajectory
    planner.MaxConnectionDistance = 0.5;
    planner.ValidationDistance = planner.MaxConnectionDistance / 2;

    % Show the place configuration with the robot in rigid body tree
    % environment with goal frame
    figure(f1);
    hold on
    show(goalFrame, gca);
    drawnow;

    % Compute path, short path, and interpolated path for the place
    % path = plan(planner, interpConfigurations(end, :), configSoln);
    % shortPath = shorten(planner, path, numIter);
    % interpConfigurations = interpolate(planner, shortPath);
    % cumulativePath{3} = interpConfigurations;

    % % Delete Goal Region from the figure
    % delete(findobj(f1, 'type', 'hgtransform'));
    % hold on;
    % show(goalFrame, gca);

    % % Show animation of the interpolated configuration for place trajectory
    % if showAnimation
    %     rateObj = rateControl(60);
    %     for i = 1 : size(interpConfigurations, 1)
    %         show(ur5e, interpConfigurations(i, :), 'PreservePlot', false, 'Frames', 'off', 'Collisions', 'off', 'Visuals', 'on', 'FastUpdate', true);
    %         waitfor(rateObj);
    %     end
    % end

    % Define velocity and acceleration limits
    maxqd = pi/2; % in rad/s
    maxqdd = deg2rad(100); % in rad/s^2
    vellimits = repmat([-maxqd; maxqd], 1, 6);
    accellimits = repmat([-maxqdd; maxqdd], 1, 6);

    % Rest position
    % Send robot to home position after placing the object
    % Update the Planner
    planner = manipulatorRRT(ur5e, env);
    planner.MaxConnectionDistance = 1;
    planner.ValidationDistance = 0.5;
    planner.SkippedSelfCollisions = 'parent';

    % Define retract position 100mm above the placed element
    currentPose = getTransform(ur5e, interpConfigurations(end, :), ur5e.BodyNames{end});
    retractPose = currentPose;
    retractPose(3,4) = retractPose(3,4) + 0.1; % Add 100mm in Z direction

    % Compute inverse kinematics for retract pose
    ik = inverseKinematics('RigidBodyTree', ur5e);
    weights = [1, 1, 1, 1, 1, 1];
    [configRetract, solnInfo] = ik(ur5e.BodyNames{end}, retractPose, weights, interpConfigurations(end, :));

    if solnInfo.Status == "success"
        path = plan(planner, interpConfigurations(end, :), configRetract);
        shortPath = shorten(planner, path, numIter);
        interpConfigurations = interpolate(planner, shortPath);

        % Plan path from retract position to home position
        pathToHome = plan(planner, configRetract, homePosition);
        shortPathToHome = shorten(planner, pathToHome, numIter);
        interpConfigurations = [interpConfigurations; interpolate(planner, shortPathToHome)];
    else
        disp('Failed to find IK solution for retract pose.');
    end

    % Interpolate the rest trajectory using contopptraj
    [q, qd, qdd, t] = contopptraj(interpConfigurations', vellimits', accellimits', NumSamples=size(interpConfigurations, 1) * 10);

    % Animate the movement back to home position
    if showAnimation
        rateObj = rateControl(60);
        for i = 1 : size(q, 2)
            show(ur5e, q(:, i)', 'PreservePlot', false, 'Frames', 'off', 'Collisions', 'off', 'Visuals', 'on', 'FastUpdate', true);
            
            endEffectorName = 'Bellow';
            endEffectorPose = getTransform(ur5e, q(:, i)', 'Bellow');
            
            % Ekstrakcja pozycji (x, y, z)
            position = endEffectorPose(1:3, 4);
            x = position(1);
            y = position(2);
            z = position(3);
            
            % Ekstrakcja kąta obrotu wokół osi Z
            R = endEffectorPose(1:3, 1:3);
            theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

            currentTime = toc(simStartTime);

            timeLog = [timeLog; currentTime];
            xLog = [xLog; x];
            yLog = [yLog; y];
            zLog = [zLog; z];
            thetaLog = [thetaLog; theta];

            waitfor(rateObj);
        end
    end
    

    %% Trajectory interpolation using contopptraj with the max acceleration and max velocity bounds
    % Robot Parameters
    maxqd = pi/2; % in rad/s
    maxqdd = deg2rad(100); % in rad/s2
    vellimits = repmat([-maxqd; maxqd],1,6);
    accellimits  = repmat([-maxqdd; maxqdd],1,6);

    % Interpolate the approach trajectories
    path1 = cumulativePath{1};
    [q,qd,qdd,t] = contopptraj(path1',vellimits',accellimits',NumSamples=size(path1,1)*10);

    % Show animation of final interpolated trajectory if flag is enabled
    if showAnimationTraj
        hold on;
        rateObj = rateControl(60);
        for i = 1 : size(q,2)
            show(ur5e,q(:,i),'PreservePlot',false,'Frames','off','Collisions','off','Visuals','on','FastUpdate',true);
            
            endEffectorName = 'Bellow';
            endEffectorPose = getTransform(ur5e, q(i,:), 'Bellow');
            
            % Ekstrakcja pozycji (x, y, z)
            position = endEffectorPose(1:3, 4);
            x = position(1);
            y = position(2);
            z = position(3);
            
            % Ekstrakcja kąta obrotu wokół osi Z
            R = endEffectorPose(1:3, 1:3);
            theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

            currentTime = toc(simStartTime);

            timeLog = [timeLog; currentTime];
            xLog = [xLog; x];
            yLog = [yLog; y];
            zLog = [zLog; z];
            thetaLog = [thetaLog; theta];
            
            waitfor(rateObj);
        end
    end

    % Interpolate retract + place trajectory (Note that this is combine
    % trajectory)
    path21 = cumulativePath{2};
    % path22 = cumulativePath{3};

    % Normalized distance in joint space
    dist_joints_1 = norm(cumulativePath{2}(1,:)-cumulativePath{2}(end,:));
    % dist_joints_2 = norm(cumulativePath{3}(1,:)-cumulativePath{3}(end,:));
    dist_total = dist_joints_1; %+ dist_joints_2;

    % Combine both trajectory
    % path2=[path21;path22];
    path2=path21;
    initialGuessPath2 = [linspace(0,dist_joints_1/dist_total,size(path21,1))]% linspace(dist_joints_1/dist_total,1,size(path22,1))];

    % Remove Duplicate Rows
    path2(size(path21,1),:) = [];
    initialGuessPath2(size(path21,1)) = [];

    % Compute piecewise polynomial
    pp = interp1(initialGuessPath2,path2,'spline','pp');

    % Apply contopptraj for retract + place trajectory
    [q,qd,qdd,t] = contopptraj(pp,vellimits',accellimits',NumSamples=size(path2,1)*10);

    % Show animation of the trajectory
    if showAnimationTraj
        hold on;
        rateObj = rateControl(60);
        for i = 1 : size(q,2)
            show(ur5e,q(:,i),'PreservePlot',false,'Frames','off','Collisions','off','Visuals','on','FastUpdate',true);
            
            endEffectorName = 'Bellow';
            endEffectorPose = getTransform(ur5e, q(:,i), 'Bellow');
            
            % Ekstrakcja pozycji (x, y, z)
            position = endEffectorPose(1:3, 4);
            x = position(1);
            y = position(2);
            z = position(3);
            
            % Ekstrakcja kąta obrotu wokół osi Z
            R = endEffectorPose(1:3, 1:3);
            theta = atan2(R(2,1), R(1,1)); % Kąt obrotu wokół osi Z

            currentTime = toc(simStartTime);

            timeLog = [timeLog; currentTime];
            xLog = [xLog; x];
            yLog = [yLog; y];
            zLog = [zLog; z];
            thetaLog = [thetaLog; theta];
            
            waitfor(rateObj);
        end
    end

    % Delete Part Ground Truth
    % partGT(partID,:) = [];
end


% Po zakończeniu pętli symulacji, wykreślenie danych
figure;

subplot(4,1,1);
plot(timeLog, xLog, 'LineWidth', 1.5);
xlabel('Czas [s]');
ylabel('X [m]');
title('Pozycja X końcówki manipulatora');

subplot(4,1,2);
plot(timeLog, yLog, 'LineWidth', 1.5);
xlabel('Czas [s]');
ylabel('Y [m]');
title('Pozycja Y końcówki manipulatora');

subplot(4,1,3);
plot(timeLog, zLog, 'LineWidth', 1.5);
xlabel('Czas [s]');
ylabel('Z [m]');
title('Pozycja Z końcówki manipulatora');

subplot(4,1,4);
plot(timeLog, thetaLog, 'LineWidth', 1.5);
xlabel('Czas [s]');
ylabel('\theta [rad]');
title('Kąt obrotu wokół osi Z końcówki manipulatora');

% Dostosowanie układu wykresów
sgtitle('Zmiany pozycji i orientacji końcówki manipulatora w czasie');  % Tytuł całego układu wykresów

% Save data to CSV file
data = table(timeLog, xLog, yLog, zLog, thetaLog);
writetable(data, 'logData.csv');