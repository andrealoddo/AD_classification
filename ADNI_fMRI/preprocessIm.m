function I = preprocessIm(filename, scale)
    
    I = imread( filename );

    % resize image to fit CNN's input layer
    I = imresize( I, scale );
    
    if ismatrix(I)
        I = cat(3,I,I,I);
    end
end

