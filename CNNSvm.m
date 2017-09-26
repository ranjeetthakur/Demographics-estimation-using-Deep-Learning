%% Train a classifier using extracted features 
trainingLabels = trainingSet.Labels;

% Here I train a linear support vector machine (SVM) classifier.
svmmdl = fitcsvm(trainingFeatures' ,trainingLabels);

% Perform cross-validation and check accuracy
cvmdl = crossval(svmmdl,'KFold',10);
fprintf('kFold CV accuracy: %2.2f\n',1-cvmdl.kfoldLoss)

