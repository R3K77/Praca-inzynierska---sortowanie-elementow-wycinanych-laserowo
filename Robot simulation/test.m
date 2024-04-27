clc
clear
close all

robot = loadrobot("kukaIiwa7","DataFormat","row","Gravity",[0 0 -9.81])
currentRobotJConfig = homeConfiguration(robot);

% robot.show(currentRobotJConfig)
% robot.inverseDynamics

% Get the end effector position
endEffectorPosition = getTransform(robot, currentRobotJConfig, 'iiwa_link_ee_kuka')

% Define the start and end configurations
startConfig = currentRobotJConfig;
endConfig = homeConfiguration(robot);

env = {collisionMesh([0.0 0.0 0.0; 1.0 0.0 0.0; 1.0 0.5 0.0; 1.0 0.0 0.0])};
env{1}.Pose(1:3, end) = [0.0, 0.0, 0.0];

% Define the time span for the trajectory
timeSpan = 0:3 % Specify the start and end time for the trajectory

% Calculate waypoints through startConfig to endConfig
numWaypoints = 10; % Specify the number of waypoints
% waypoints = linspace(startConfig, endConfig, numWaypoints); % Generate the waypoints
waypoints = [0 0 1; 0.1 0 1; 0.2 0 1; 0.3 0 1]

% Generate the trajectory
trajectory = waypointTrajectory(waypoints, timeSpan);

% Visualize the trajectory
figure;

for i = 1:1000
    show(robot, [0 i*0.01 1 0 0 0 0]);
    hold on
    show(env{1})
    hold off
    drawnow;

end


% 

% 
% 
% show(robot); % Show the robot
% hold on;
% show(env{1}); % Show the obstacle
% 
% rrt = manipulatorRRT(robot,env);
% rrt.SkippedSelfCollisions = "parent";
% 
% startConfig = [0.08 -0.65 0.05 0.02 0.04 0.49 0.04];
% goalConfig =  [2.97 -1.05 0.05 0.02 0.04 0.49 0.04];
% 
% rng(0)
% path = plan(rrt,startConfig,goalConfig);
% 
% interpPath = interpolate(rrt,path);
% clf
% for i = 1:20:size(interpPath,1)
%     show(robot,interpPath(i,:));
%     hold on
% end
% 
% hold off