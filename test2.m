frameNumber = 0;
vr = VideoReader('Resume.MP4');
opticFlow = opticalFlowFarneback;
imageSize = [227 227];
vp = vision.VideoPlayer;

%%
while hasFrame(vr)
    % Count frames
    frameNumber = frameNumber + 1
    
    % Step 1. Read Frame
	vFrame = readFrame(vr);
    
    % Step 2. Detect ROI
    frameGray = rgb2gray(vFrame);            % Convert to gray for detection
    [bboxes, flow] = findPet(frameGray,opticFlow);   % Find bounding boxes
        
    if ~isempty(bboxes)
        img = zeros([imageSize 3 size(bboxes,1)]);
        for ii = 1:size(bboxes,1)
            img(:,:,:,ii) = imresize(imcrop(vFrame,bboxes(ii,:)),imageSize(1:2));
            
            % Step 3: Extract image features and predict label
            imageFeatures = activations(convnet, img(:,:,:,ii), featureLayer);
            label = predict(svmmdl, imageFeatures);

            % Step 4: Annotation
            vFrame = insertObjectAnnotation(vFrame,'Rectangle',bboxes(ii,:),cellstr(label),'FontSize',14);
            cellstr(label)
        end
        step(vp, vFrame);
        
    end
    
end
