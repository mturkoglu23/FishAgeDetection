clc;clear;
load path_left.mat
load label_left.mat
load path_right.mat
load label_right.mat

layer2 ='fc6';
layer6='predictions';
layer7='conv53';

net4 =vgg16;
net6=inceptionresnetv2;
net7=darknet53;

data_left=path_left;
filename='C:\Users\...';
m=1;
n=1;
labb_left=double(label_left);
for i=1:4109
    i
    index=num2str(data_left(i,1));

        img=imread(strcat(filename, num2str(data_left(i,1))));
        aa=size(img);
            if length(aa)==2
                img=cat(3,img,img,img);
            end
        img=imresize(img,[224 224]);
        img2=imresize(img,[256 256]);
        img3=imresize(img,[299 299]);

        vgg16_Feats_left(:,i)=activations(net4,img,layer2);
        incepres_Feats_left(:,i)=activations(net6,img3,layer6);
        dark_Feats_left(:,i)=activations(net7,img2,layer7);

end

%%
data_right=path_right;
filename='C:\Users\....\';
m=1;
n=1;
labb_right=double(label_right);
for i=1:4109
    i
    index=num2str(data_right(i,1));

        img=imread(strcat(filename, num2str(data_right(i,1))));
        aa=size(img);
            if length(aa)==2
                img=cat(3,img,img,img);
            end

        img=imresize(img,[224 224]);
        img2=imresize(img,[256 256]);
        img3=imresize(img,[299 299]);

        vgg16_Feats_right(:,i)=activations(net4,img,layer2);
        incepres_Feats_right(:,i)=activations(net6,img3,layer6);
        dark_Feats_right(:,i)=activations(net7,img2,layer7);

end

%%

total_number = 4109;
percent=0.96;
tt=round(percent*total_number);
ff_right=[vgg16_Feats_right;dark_Feats_right;incepres_Feats_right];
ff_left=[vgg16_Feats_left;dark_Feats_left;incepres_Feats_left];

WP1 =[ff_right;ff_left]';  
WT=[labb_left];
fea=[WP1,WT];

[rr1,ww]=relieff(WP1,WT,10);
for i=1:1000
    yoz1(:,i)=WP1(:,rr1(i));
end
% 
WP=yoz1;

% [rr]=randperm(total_number);
Trn(1:tt,:)   = WP(rr(1:tt),:);
Trn_label(1:tt,:)   = WT(rr(1:tt),:);

Test= WP(rr(tt+1:end),:);
Tst_label= WT(rr(tt+1:end),:);

featuresTrain=Trn;
YTrain=Trn_label;
featuresTest=Test;
YTest=Tst_label;

regressionGP = fitrgp(featuresTrain,YTrain,...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'rationalquadratic', ...
    'Standardize', true);

YPred = predict(regressionGP,featuresTest);
err = immse(double(YPred),YTest)


