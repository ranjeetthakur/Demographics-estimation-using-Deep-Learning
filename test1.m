newImage = '/home/ranjeet/ranjeet/Datasets/hijab.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')


newImage = '/home/ranjeet/ranjeet/Datasets/hijab1.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')



newImage = '/home/ranjeet/ranjeet/Datasets/hijab2.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')


newImage = '/home/ranjeet/ranjeet/Datasets/hijab3.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')



newImage = '/home/ranjeet/ranjeet/Datasets/hijab4.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')



newImage = '/home/ranjeet/ranjeet/Datasets/hijab5.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')


newImage = '/home/ranjeet/ranjeet/Datasets/hijab6.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(svmmdl, imageFeatures) 
figure()
imshow(newImage)
annotation('textbox',[0.15,0.65,0.3,0.15],'String',string(label),'FontSize',44,'Color','r')

