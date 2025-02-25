function [RefPose, partID, goalZoffset] = selectPart(partGT,goalPoints)
    persistent partCounter;  % Counter to keep track of the current part
    if isempty(partCounter)
        partCounter = 1;
    else
        partCounter = partCounter + 1;
    end
    
    % Ensure the partCounter does not exceed the number of parts available
    % if partCounter > size(partGT, 1)
    %     error('All parts have been selected.');
    % end
    
    % partID = partCounter;
    
    % Ensure the STL file for the current partID exists
    % stlFile = sprintf('meshes/output_%d.stl', partID - 1);

    data = readtable('element_details.csv');

    filename = data.Nazwa{partCounter};
    % filename_dir = strcat('meshes/', filename);

    % partID - indeks elementu w pliku CSV
    filename = strrep(filename, '.stl', '');
    partID = findStringInCSV('element_details.csv', filename) - 1;

    % if ~isfile(filename_dir)
    %     error('STL file for partID %d does not exist.', partID);
    % end
    
    goalZoffset = 0.01;
    
    % Set the reference pose based on the current part's goal points
    RefPose = [goalPoints(partID, 1) + 0.2, goalPoints(partID, 2), partGT(partID, 3) + 0.02];
end
