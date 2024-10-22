function lineNumberFound = findStringInCSV(filename, searchString)
    % Otwórz plik CSV
    fid = fopen(filename, 'r');

    lineNumberFound = [];
    
    if fid == -1
        error('Nie można otworzyć pliku: %s', filename);
    end
    
    % Zmienna do przechowywania numeru wiersza
    lineNumber = 0;
    
    % Przechodzenie przez plik linia po linii
    while ~feof(fid)
        % Czytaj linię
        line = fgetl(fid);
        lineNumber = lineNumber + 1;
        
        % Sprawdź, czy linia zawiera szukany string
        if contains(line, searchString)
            fprintf('String "%s" znaleziony w linii %d: %s\n', searchString, lineNumber, line);
            lineNumberFound = lineNumber;
        end
    end
    
    % Zamknij plik po zakończeniu przetwarzania
    fclose(fid);
end
