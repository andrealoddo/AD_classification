function imds = prepareDatastore(path, classes)
            
    imgPaths = [];
    labels = [];
    
    for c = 1:numel(classes)
        
        imgSets = imageSet(fullfile(path, classes{c}), 'recursive');

        for i = 1:numel(imgSets)
            imgPaths = [ imgPaths; imgSets(i).ImageLocation' ];
            class_labels = strings( numel(imgSets(i).ImageLocation'),1 );
            class_labels(:) = classes{c};
            labels = [ labels; class_labels ];
        end
        
    end
    
    %labels = cellstr(labels);
    labels = categorical(labels);

    imds = imageDatastore(imgPaths);
    imds.Labels = labels;


end