% extract and concatenate deep features
alexnet_fc = 'fc7';
inc_fc = 'predictionsawdwq';

net = open('C:\Users\loand\Documents\GitHub\Tesisti\AD_classification\ADNI_fMRI\models\alexnet___EP4__MBS16__AUG0.mat');
alexnet = net.trainedNet;

net = open('C:\Users\loand\Documents\GitHub\Tesisti\AD_classification\ADNI_fMRI\models\resnet101__EP1__MBS8__AUG0.mat');
resnet = net.trainedNet;

net = open('C:\Users\loand\Documents\GitHub\Tesisti\AD_classification\ADNI_fMRI\models\inceptionresnetv2__EP1__MBS8__AUG0.mat');
inception = net.trainedNet;

imds = open('imds.mat');
imds = imds.imds;

%imdsTrain = open('imdsTrain.mat');
%imdsValid = open('imdsValid.mat');
%imdsTest = open('imdsTest.mat');

labels = imds.Labels;

inputSize = alexnet.Layers(1).InputSize(:, 1:2);
imds.ReadFcn = @(filename)preprocessIm(filename, inputSize);
featuresAlex = activations(alexnet, imds, alexnet_fc, 'MiniBatchSize', 32);
featuresAlex = squeeze(featuresAlex);
featuresAlex = featuresAlex';

inputSize = resnet.Layers(1).InputSize(:, 1:2);
imds.ReadFcn = @(filename)preprocessIm(filename, inputSize);
featuresRes = activations(resnet, imds, inc_fc, 'MiniBatchSize', 32);
featuresRes  = squeeze(featuresRes );
featuresRes  = featuresRes';

inputSize = inception.Layers(1).InputSize(:, 1:2);
imds.ReadFcn = @(filename)preprocessIm(filename, inputSize);
featuresInc = activations(inception, imds, inc_fc, 'MiniBatchSize', 32);
featuresInc  = squeeze(featuresInc );
featuresInc  = featuresInc';

featuresCombined = [featuresAlex; featuresRes; featuresInc];

