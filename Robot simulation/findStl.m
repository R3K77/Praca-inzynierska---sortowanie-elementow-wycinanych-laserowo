function [plikLista, plikWybrany] = findStl(wzorzec, indeks)
    % SZUKA WSZYSTKICH PLIKÓW PASUJĄCYCH DO WZORCA I ZWRACA LISTĘ ORAZ PLIK O DANYM INDEKSIE
    %
    % Argumenty:
    % wzorzec - wzorzec pliku (np. '*.txt' lub 'data*.csv')
    % indeks - indeks pliku do zwrócenia
    %
    % Zwraca:
    % plikLista - lista wszystkich plików pasujących do wzorca
    % plikWybrany - plik znajdujący się na pozycji 'indeks' (jeśli istnieje)

    % Pobieranie listy plików pasujących do wzorca
    pliki = dir(wzorzec);
    
    % Tworzenie listy nazw plików
    plikLista = {pliki.name};
    
    % Sprawdzanie, czy lista nie jest pusta
    if isempty(plikLista)
        error('Brak plików pasujących do wzorca: %s', wzorzec);
    end
    
    % Sprawdzanie, czy indeks jest prawidłowy
    if indeks > numel(plikLista) || indeks < 1
        error('Indeks %d wykracza poza zakres. Dostępnych plików: %d', indeks, numel(plikLista));
    end
    
    % Zwracanie wybranego pliku
    plikWybrany = plikLista{indeks};
end
