function ur5e = exampleHelperAttachPart(ur5e,rotation)
  % part Dimenssions
    partwidth = 0.0508; % 2 inch
    partheight = 0.0508; % 2 inch
    partLength = 0.1016; % 4 inch

    tf_collision = eul2tform([rotation+pi/2 0 0]);

    box = [partLength partwidth partheight];
    tf_collision(:,4) = [0; 0; partheight/2; 1];

    % Trandform for the visual mesh
    tf_visual = trvec2tform([0 0 0]) * eul2tform([rotation 0 0], 'ZYX');

    % Attach collision box to the rigid body model
    transformPart = eul2tform([0 0 0]);
    transformPart(:,4) = [0; 0; 0.02; 1]; % To avoid self collision

    part = rigidBody('part');
    addCollision(part,'box',box,tf_collision);

    addVisual(part,"Mesh",strcat('meshes',filesep,'box_part.STL'),tf_visual);

    partJoint = rigidBodyJoint('partJoint','fixed');
    part.Joint = partJoint;
    setFixedTransform(part.Joint, transformPart);
    curEndEffectorBodyName = ur5e.BodyNames{14};
    addBody(ur5e,part,curEndEffectorBodyName);
end