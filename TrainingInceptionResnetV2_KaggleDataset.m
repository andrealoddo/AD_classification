% Classificazione livello di demenza (Dataset: kaggle)
% Multiclasse: NonDemented, VeryMildDemented, MildDemented,
% ModerateDemented

%-----------------------  Gestione dataset --------------------------
% Vengono estratte le immagini dalla cartella "all image", e per ogni
% immagine si controlla il nome della stessa: la parte iniziale del nome
% rivela la classe di appartenenza di quella immagine. Ogni immagine viene
% salvata nell'array brainImgs e l'etichetta corrispondente nell'array
% brainLabels.
filepath1 = "Alzheimer_s Dataset\all image";
S = dir(fullfile(filepath1,"*.jpg"));
for i = 1: numel(S)
    brainImgs(i) = fullfile(filepath1,S(i).name);
    if(contains(S(i).name,"verymildDem"))
               brainLabels(i) = "very mild dementia";
    else
        if(contains(S(i).name,"nonDem"))
           brainLabels(i) = "non dementia";        
        else
            if(contains(S(i).name,"mildDem"))
        brainLabels(i) = "mild dementia";
            else
                if(contains(S(i).name,"moderateDem"))
                   brainLabels(i) = "moderate dementia";
                end
            end
        end
    end
end

% ------------------   Creazione del datastore   -----------------------
brainImgsDs = imageDatastore(brainImgs); % Datastore 
brainLabels = categorical(brainLabels);
brainImgsDs.Labels = brainLabels; % Etichette del datastore

% Divisione tra training set(80%),validation set (10%) e test set (10%)
[trainImgs,valSet,testImgs] = splitEachLabel(brainImgsDs,0.8,0.1,0.1,'randomized');
testImgs.ReadFcn = @(filename)gray2rgb_resize(filename,299);  %Vengono ridimensionate le immagini del test
% Viene applicata l'augmentation al training set per aumentare la diversitÓ
% all'interno del dataset
imageAugmenter = imageDataAugmenter("RandRotation",[-35 35],"RandXScale",[0.5 4],"RandYScale",[0.5 1]);
trainAug = augmentedImageDatastore([299 299],trainImgs,"ColorPreprocessing","gray2rgb","DataAugmentation",imageAugmenter);
valAug = augmentedImageDatastore([299 299],valSet,"ColorPreprocessing","gray2rgb","DataAugmentation",imageAugmenter);


% ------------------   Creazione della rete   --------------------------
net = inceptionresnetv2;
layers = layerGraph(net);
% % Modifico il terzultimo e l'ultimo strato della rete
newFCLayer = fullyConnectedLayer(4,'Name','new_fc'); 
layers = replaceLayer(layers,'predictions',newFCLayer);
newCLLayer = classificationLayer('Name','new_output');
layers = replaceLayer(layers,'ClassificationLayer_predictions',newCLLayer);


% Vengono settate le training options
options = trainingOptions('adam',"Plots","training-progress","ValidationData",valAug,"InitialLearnRate",0.0001,"MaxEpochs",30,"ValidationFrequency",30,"MiniBatchSize",10);
% -------------------    Train Network    -------------------------------
trainedNet = trainNetwork(trainAug,layers,options);
preds = classify(trainedNet, testImgs);
accuracy = nnz(preds == testImgs.Labels)/numel(preds)

%Confusion Chart
chart = confusionchart(preds,testImgs.Labels)


