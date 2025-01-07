function outputRobot = addGripper(inputRobot)


    % Add I/O Coupling
    % coupling = rigidBody('IO_Coupling');
    % addVisual(coupling,"Mesh",strcat('meshes',filesep,'IO_coupling_MW.STL'),eul2tform([pi 0 0]));
    % tf = eul2tform([0 0 0]);
    % tf(:,4) = [0; 0; 0.006; 1]; % To avoid collision
    % % addCollision(coupling,'cylinder',[0.0325 0.010],tf);
    % couplingJoint = rigidBodyJoint('couplingJoint','fixed');
    % coupling.Joint = couplingJoint;
    % curEndEffectorBodyName = inputRobot.BodyNames{10};
    % addBody(inputRobot,coupling,curEndEffectorBodyName);
    
    % % Add Gripper Unit
    % transformGripper = eul2tform([0 0 0]);
    % transformGripper(:,4) = [0; 0; 0.0139; 1]; % The width is 16.9 or 13.9
    % gripper = rigidBody('EPick');
    % tf = eul2tform([0 0 pi/2]);
    % tf(:,4) = [0; 0; 0.101-0.0139; 1]; % Gripper Width
    % addVisual(gripper,"Mesh",strcat('meshes',filesep,'Epick_MW_2.STL'),tf);
    % tf = eul2tform([0 0 0]);
    % tf(:,4) = [0; 0; 0.045; 1]; % Gripper Width
    % addCollision(gripper,'cylinder',[0.083/2 0.1023],tf);
    % gripperJoint = rigidBodyJoint('gripperJoint','fixed');
    % gripper.Joint = gripperJoint;
    % setFixedTransform(gripper.Joint, transformGripper);
    % curEndEffectorBodyName = inputRobot.BodyNames{10};
    % addBody(inputRobot,gripper,curEndEffectorBodyName);
    
    
    % Add Extention tube
    transformTube = eul2tform([0 0 0]);
    transformTube(:,4) = [0; 0; 0.15; 1]; % The width is 101
    tube = rigidBody('Tube');
    tf = eul2tform([0 0 0]);
    tf(:,4) = [0; 0; -0.15; 1]; % Gripper Width [0; 0; -0.007; 1]
    addVisual(tube,"Mesh",strcat('meshes',filesep,'extention_tube copy.stl'),tf);
    % tf = eul2tform([0 0 0]);
    % tf(:,4) = [0; 0; 0.020; 1]; % to avoid collision
    % addCollision(tube,'cylinder',[0.005 0.020],tf);
    tubeJoint = rigidBodyJoint('tubeJoint','fixed');
    tube.Joint = tubeJoint;
    setFixedTransform(tube.Joint, transformTube);
    curEndEffectorBodyName = inputRobot.BodyNames{10};
    addBody(inputRobot,tube,curEndEffectorBodyName);
    
    
    % % Add Bellow Small
    transformBellow = eul2tform([0 0 0]);
    transformBellow(:,4) = [0; 0; 0.040; 1]; % The width is 101
    bellow = rigidBody('Bellow');
    tf = eul2tform([0 0 -pi/2]);
    tf(:,4) = [-0.030; 0; 0; 1]; % Gripper Width
    addVisual(bellow,"Mesh",strcat('meshes',filesep,'suction_cup.stl'),tf);
    bellowJoint = rigidBodyJoint('bellowJoint','fixed');
    bellow.Joint = bellowJoint;
    setFixedTransform(bellow.Joint, transformBellow);
    curEndEffectorBodyName = inputRobot.BodyNames{11};
    addBody(inputRobot,bellow,curEndEffectorBodyName);
    
    outputRobot = inputRobot;
end