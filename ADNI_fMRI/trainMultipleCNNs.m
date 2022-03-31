% 6 Class classification
%ADNI_fMRI = readtable('C:\Users\loand\Pictures\ImmaginiLavoro\Medical\Alzheimer\ADNI-f\ADNI2F.csv', ...
%    'PreserveVariableNames', true);

% ----------------------- GLOBAL SETTINGS -----------------------
AUG = 0;
classNumber = 3;
%cnns = {alexnet, vgg16, vgg19, resnet18, resnet50, resnet101, googlenet, inceptionv3, shufflenet, squeezenet, mobilenetv2};
%cnnNames = {'alexnet', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'resnet101', 'googlenet', 'inceptionv3', 'shufflenet', 'squeezenet', 'mobilenetv2'};

cnns = {inceptionresnetv2};
cnnNames = { 'inceptionresnetv2'};


% ----------------------- FOLDERS -----------------------
if ispc
    % Windows dataset path
    rootPath = 'C:\Users\loand\Pictures\ImmaginiLavoro\Medical\Alzheimer\ADNI-f\IMG2';
else
    % WS dataset path
    rootPath = '/home/server/MATLAB/dataset/ALL_IDB/ALL_IDB2';
end

modelsPath = 'models';
cmPath = 'cm';
checkPath = 'checkpoints';

classes = dir(rootPath);
classes = {classes(3:end).name}.';

if exist('datastore2.mat', 'file') ~= 2
    tic
    imds = prepareDatastore(rootPath, classes);
    toc
else
    load('datastore2.mat');
end


%----------------------- DATASET ACQUISITION and OVERSAMPLING -----------------------

% Train + validation sets with split
[imdsTrain, imdsValid, imdsTest] = splitEachLabel( imds, 0.3, 0.2, 0.5 );

% Check label frequency for imbalance
labelCount = countEachLabel(imdsTrain);
%histogram(imdsTrain.Labels);title('label frequency')

% Perform training set oversampling to avoid class imbalance
%imdsTrain = overSampling(imdsTrain);


%----------------------- TESTING WITH CNNS -----------------------
for i = 1:numel(cnns)

    % Load the network
    net = cnns{i};
    netName = cnnNames{i};
    netCheckPath = fullfile(checkPath, netName);
    
    % Augmentation
    if(AUG == 1)
        augmenter = imageDataAugmenter( ...
            'RandRotation',[0 360], ...
            'RandXReflection', 1, ...
            'RandYReflection', 1, ...
            'RandXShear', [-0.05 0.05], ...
            'RandYShear', [-0.05 0.05]);

        imdsTrain = augmentedImageDatastore(net.InputSize, imdsTrain, 'DataAugmentation', augmenter);
        imdsValid = augmentedImageDatastore(net.InputSize, imdsValid, 'DataAugmentation', augmenter);
    end
    
    [~, ~] = trainCNN(net, netName, netCheckPath, modelsPath, cmPath, ...
        AUG, classNumber, imdsTrain, imdsValid, imdsTest);
    
end

