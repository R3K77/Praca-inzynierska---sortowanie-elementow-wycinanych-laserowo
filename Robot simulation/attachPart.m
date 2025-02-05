function ur5e = attachPart(ur5e, rotation, partID, collisionPoint, visualPoint)
  % Part Dimensions
  data = readtable('element_details.csv');
  filename = data.Nazwa{partID};
  
  filename = strcat('meshes/', filename, '.stl');
  FV = stlread(filename);
  partDimensions = max(FV.Points) - min(FV.Points);
  
  partwidth = partDimensions(2);
  partheight = partDimensions(3);
  partLength = partDimensions(1);
  
  % Transform for the collision box
  tf_collision = eul2tform([rotation+pi/2 0 0]);
  tf_collision(:, 4) = [collisionPoint(:); 1];
  
  % Transform for the visual mesh
  tf_visual = trvec2tform([visualPoint(:)]') * eul2tform([rotation 0 0], 'ZYX');
  
  % Dimensions of the box
  box = [partLength partwidth partheight];
  
  % Attach collision box to the rigid body model
  transformPart = eul2tform([0 0 0]);
  transformPart(:, 4) = [0; 0; 0; 1]; % To avoid self-collision
  
  part = rigidBody('part');
  % addCollision(part, 'box', box, tf_collision);
  
  addVisual(part, "Mesh", filename, tf_visual);
  
  partJoint = rigidBodyJoint('partJoint', 'fixed');
  part.Joint = partJoint;
  setFixedTransform(part.Joint, transformPart);
  curEndEffectorBodyName = ur5e.BodyNames{end};
  addBody(ur5e, part, curEndEffectorBodyName);
end
