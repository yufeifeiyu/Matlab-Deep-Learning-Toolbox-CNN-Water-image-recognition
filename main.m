
%---------------------------------------------------------------- 
% filename :main.m
% created by yufei at  2/14/2020
% description :The training set and test set were generated based on the images and labels, and the CNN training test was used to draw the original images, divide the images and forecast images, and output accuracy
%----------------------------------------------------------------
clear;
fname='image1.json'; %待读取的文件名称
image_name='image1.TIF';%待读取的图片名称
DataMark(fname,image_name);%数据标记函数

%读取处理后的图像数据
load image_label
image=imread(image_name);%读取原始图像
[m1,n1,k1]=size(image);%原始尺寸
label2=label;


image=imresize(image,[m,n]);%调整图像尺寸，以适合cnn的输入
%图像切分，将image和label数据切分为c*r个16*16的cell数据
c=zeros(1,m/16)+16;
r=zeros(1,n/16)+16;
image1=mat2cell(image,c,r,k);
label1=mat2cell(label,c,r);

%图像标记，如果一块中1的数量大于一半，认为是水体标记为1，反之认为是陆地标记为0
label=zeros(m*n/16/16,k);
for i=1:m/16
    for j=1:n/16
        if(sum(sum(label1{i,j})))>128
            label((i-1)*m/16+j)=1;
        else
            label((i-1)*m/16+j)=0;
        end
    end
end


%将image1的cell数据转为4D-array数据，以作为cnn的输入
input=zeros(16,16,k,m*n/16/16);
for i=1:m/16
    for j=1:n/16
        input(:,:,:,(i-1)*m/16+j)=image1{i,j};
    end
end
output=categorical(label);%把double型的label数据转为cnn所用的categorical型数据

train_input=input(:,:,:,1:floor(m*n*0.8/16/16));%取80%的输入样本作为训练集输入
test_input=input(:,:,:,ceil(m*n*0.8/16/16):m*n/16/16);%取20%的输入样本作为测试集输入
train_output=output(1:floor(m*n*0.8/16/16));%取80%的输出样本作为训练集输出
test_output=output(ceil(m*n*0.8/16/16):m*n/16/16);%取20%的输出样本作为测试集输出


%设计cnn
%九层卷积神经网络
%1.输入层，数据大小16*16*k，k为图像通道数。
%2.卷积层，16个3*3大小的卷积核，步长为1，对边界补0。
%3.池化层，使用2*2的核，步长为2。
%4.卷积层，32个3*3大小的卷积核，步长为1，对边界补0。
%5.池化层，使用2*2的核，步长为2。
%6.卷积层，64个3*3大小的卷积核，步长为1，对边界补0。
%7.池化层，使用2*2的核，步长为2。
%8.全连接层，30个神经元。
%9.全连接层，2个神经元。
layers = [
    imageInputLayer([16 16 k])%输入层，k为通道数
    
    convolution2dLayer(3,16,'Padding','same')%卷积层16个3*3卷积核
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)%池化层2*2，步长2
    
    convolution2dLayer(3,32,'Padding','same')%卷积层32个3*3卷积核
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)%池化层2*2，步长2
    
    convolution2dLayer(3,64,'Padding','same')%卷积层64个3*3卷积核
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)%池化层2*2，步长2
    
    fullyConnectedLayer(30)%30个节点的全连接层
    fullyConnectedLayer(2)%2个节点的全连接层
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...%学习速率
    'MaxEpochs',10-ceil(m/1024)*2, ...%最大迭代次数
    'Shuffle','every-epoch', ...
    'L2Regularization',0.001,...%L2正则化参数
    'Verbose',false, ...
    'Plots','training-progress');
[net,info] = trainNetwork(train_input,train_output,layers,options);%训练网络
 
%     'ValidationData',{test_input,test_output},...%防止过拟合
%      'ValidationFrequency',10, ...
     
YPred = classify(net,test_input);%测试网络
if size(YPred)~=size(test_output)
    test_output=test_output';
end
accuracy = sum(YPred == test_output)/numel(test_output)%输出测试集的ac率

YPred = classify(net,input);%把全部输入放进网络进行预测，返回预测结果可视化
out_image=zeros(m/16,n/16);%out_image为最后预测图片
for i=1:m/16 %将categorical类型数据转回double型
    for j=1:n/16
        out_image(i,j)=YPred((i-1)*m/16+j);
        out_image(i,j)=out_image(i,j)-1;
    end
end
%展示原始图像、划分后图像和预测图像
subplot(2,2,1),imshow(imresize(image,[m1,n1])),title('原始图像');
%正常 如果标记时取的水体就是正常
subplot(2,2,2),imshow(imresize(label2,[m1,n1])),title('划分后图像');
out_image=imresize(out_image,[m1,n1]);%将图片大小重置为原始大小

%取反 如果标记时取的陆地就是取反
% subplot(2,2,2),imshow(imresize(1-label2,[m1,n1])),title('划分后图像');
% out_image=imresize(1-out_image,[m1,n1]);%将图片大小重置为原始大小
subplot(2,2,3),imshow(out_image),title('预测图像');
