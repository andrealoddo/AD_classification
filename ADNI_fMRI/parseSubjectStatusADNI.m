function status = parseSubjectStatusADNI(filepath, table)
    
    [path, name] = fileparts(filepath); % Gets name from path

    % Acquisition of labels    
    a = strsplit(name,'_'); % Splits name of file 
    column1 = a{end - 1}; % Field to look for in the first column of the table
    column2 = [a{2} '_' a{3} '_' a{4}]; % Field to look for in the second column of the table
    ix=(ismember(table{:,1},{column1}) & ismember(table{:,2},{column2})); % Looks for entry in the table
    status = table{ix,3}{1}; % Gets status of patient
end

