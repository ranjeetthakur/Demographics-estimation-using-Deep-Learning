convnet = helperImportMatConvNet('C:\Users\ranje\Desktop\CNN-Oxford\practical-cnn-2015a\imagenet-caffe-alex.mat');
convnet.Layers

dataFolder = ' C:\Users\ranje\pet_images';
categories = {'male', 'female'};
imds = imageDatastore(fullfile(dataFolder, categories), ...
    'LabelSource', 'foldernames');
tbl = countEachLabel(imds)

%% Use the smallest overlap set
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%% Pre-process Images For CNN
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
 
%% Divide data into training and testing sets
[trainingSet, testSet] = splitEachLabel(imds, 0.1, 'randomize');

% Get the network weights for the second convolutional layer
w1 = convnet.Layers(2).Weights;
 
% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 
 
% Display a montage of network weights. There are 96 individual
% sets of weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')

featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

