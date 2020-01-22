function vectorImage = imageTo20x20Gray(fileName,cropPercentage,rotStep)

Image3DmatrixRGB = imread(fileName);
Image3DmatrixYIQ = rgb2ntsc(Image3DmatrixRGB);
Image2DmatrixBW = Image3DmatrixYIQ(:,:,1);
oldSize = size(Image2DmatrixBW);
cropDelta = floor((oldSize-min(oldSize)).*(cropPercentage/100));
finalSize = oldSize - cropDelta;
cropOrigin = floor(cropDelta/2)+1;
copySize = cropOrigin + finalSize -1;
croppedImage = Image2DmatrixBW(cropOrigin(1):copySize(1),cropOrigin(2):copySize(2));

scale = [20 20]./ finalSize;
newSize = max(floor(scale.*finalSize),1); %geral

rowIndex = min(round(((1:newSize(1))-0.5)./scale(1)+0.5),finalSize(1));
colIndex = min(round(((1:newSize(2))-0.5)./scale(2)+0.5),finalSize(2));

newImage = croppedImage(rowIndex,colIndex,:);
newAlignedImage = rot90(newImage, rotStep);

%invertendo cor da imagem (NECESSÁRIO PQ O CLASSIFICADOR LÊ O DÍGITO EM
%BRANCO)

invertedImage = -newAlignedImage; %tem negativo aqui

maxValue = max(invertedImage(:));
minValue = min(invertedImage(:));
delta = maxValue - minValue;

normImage = (invertedImage - minValue)/delta;
%normImage = (normImage-0.5)*7;
contrastedImage = sigmoid((normImage-0.5)*5);
imshow(contrastedImage,[-1, 1]);
vectorImage = reshape(contrastedImage,1,newSize(1)*newSize(2));
