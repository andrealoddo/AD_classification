function imds = overSampling(imds)
    %Oversamples the image data which belongs to minor classes.

    if isa(imds,'matlab.io.datastore.ImageDatastore')
        % Extract files and labels information in training dataset
        files = imds.Files;
        labels = imds.Labels;
    else
        labels = imds;
    end
    
    %convert categorical labels into numerical ones
    [G,~] = findgroups(labels);

    %extract the number of observation per class
    numObservations = splitapply(@numel,labels,G);

    % Calculate the number of images comprising the majority class.
    desiredNumObservationsPerClass = max(numObservations);
    
    orig_ind = (1:numel(labels))';
    % Use splitandapply to random oversample the minor class
    ind = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)}, orig_ind, G);
    ind = horzcat(ind{:});
    if isa(imds,'matlab.io.datastore.ImageDatastore')
        imds.Files = files(ind);
        imds.Labels = labels(ind);
    else
        imds = ind;
    end
end

%--------------------------------------------------------------------------

function files = randReplicateFiles(files,numDesired)
    if numel(files) == numDesired
        ind = 1:numDesired;
    else
        n = numel(files);
        ind = [];
        prop = floor(numDesired/n);
        for i = 1:prop
            ind = [ind 1:n];
        end
        ind = [ind randperm(n,numDesired-(n*prop))];
    end
    files = files(ind);
end
