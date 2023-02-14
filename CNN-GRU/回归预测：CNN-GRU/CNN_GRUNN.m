%% CNN-GRU多变量回归预�?
%% 加载数据与数据集划分
%  ����"CNN-GRU"ģ��
    layers = [...
        % ��������
        sequenceInputLayer([numFeatures 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')
        % CNN������ȡ
        convolution2dLayer([FiltZise 1],32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')
        % չ����
        sequenceUnfoldingLayer('Name','unfold')
        % ƽ����
        flattenLayer('Name','flatten')
        % GRU����ѧϰ
        gruLayer(128,'Name','GRU1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        % GRU���
        gruLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
        % ȫ���Ӳ�
        fullyConnectedLayer(numResponses,'Name','fc')
        regressionLayer('Name','output')    ];

    layers = layerGraph(layers);
    layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');

%% CNNGRUѵ��ѡ��
% ����������
MiniBatchSize =24;
% ����������
MaxEpochs = 60;
% ѧϰ��
learningrate = 0.005;
% һЩ��������
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end
    options = trainingOptions( 'adam', ...
        'MaxEpochs',100, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',20, ...
        'LearnRateDropFactor',0.8, ...
        'L2Regularization',1e-3,...
        'Verbose',false, ...
        'ExecutionEnvironment',mydevice,...
        'Plots','training-progress');

% 输入数据
input =data(:,1:12)';
output=data(:,13)';
nwhole =size(data,1);
% 打乱数据�?
% temp=randperm(nwhole);
% 不打乱数据集
temp=1:nwhole;
train_ratio=0.9;
ntrain=round(nwhole*train_ratio);
ntest =nwhole-ntrain;
% 准备输入和输出训练数�?
input_train =input(:,temp(1:ntrain));
output_train=output(:,temp(1:ntrain));
% 准备测试数据
input_test =input(:, temp(ntrain+1:ntrain+ntest));
output_test=output(:,temp(ntrain+1:ntrain+ntest));
%% 数据归一�?
method=@mapminmax;
[inputn_train,inputps]=method(input_train);
inputn_test=method('apply',input_test,inputps);
[outputn_train,outputps]=method(output_train);
outputn_test=method('apply',output_test,outputps);
% 创建元胞或向量，长度为训练集大小�?
XrTrain = cell(size(inputn_train,2),1);
YrTrain = zeros(size(outputn_train,2),1);
for i=1:size(inputn_train,2)
    XrTrain{i,1} = inputn_train(:,i);
    YrTrain(i,1) = outputn_train(:,i);
end
% 创建元胞或向量，长度为测试集大小�?
XrTest = cell(size(inputn_test,2),1);
YrTest = zeros(size(outputn_test,2),1);
for i=1:size(input_test,2)
    XrTest{i,1} = inputn_test(:,i);
    YrTest(i,1) = outputn_test(:,i);
end

%% 创建混合CNN-GRU网络架构
% 输入特征维度
numFeatures  = size(inputn_train,1);
% 输出特征维度
numResponses = 1;
FiltZise = 10;
%  创建"CNN-GRU"模型
    layers = [...
        % 输入特征
        sequenceInputLayer([numFeatures 1 1],'Name','input')
        sequenceFoldingLayer('Name','fold')
        % CNN特征提取
        convolution2dLayer([FiltZise 1],32,'Padding','same','WeightsInitializer','he','Name','conv','DilationFactor',1);
        batchNormalizationLayer('Name','bn')
        eluLayer('Name','elu')
        averagePooling2dLayer(1,'Stride',FiltZise,'Name','pool1')
        % 展开�?
        sequenceUnfoldingLayer('Name','unfold')
        % 平滑�?
        flattenLayer('Name','flatten')
        % GRU特征学习
        gruLayer(128,'Name','GRU1','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        % GRU输出
        gruLayer(32,'OutputMode',"last",'Name','bil4','RecurrentWeightsInitializer','He','InputWeightsInitializer','He')
        dropoutLayer(0.25,'Name','drop3')
        % 全连接层
        fullyConnectedLayer(numResponses,'Name','fc')
        regressionLayer('Name','output')    ];

    layers = layerGraph(layers);
    layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');

%% CNNGRU训练选项
% 批处理样�?
MiniBatchSize =24;
% �?��迭代次数
MaxEpochs = 60;
% 学习�?
learningrate = 0.005;
% �?��参数调整
if gpuDeviceCount>0
    mydevice = 'gpu';
else
    mydevice = 'cpu';
end
    options = trainingOptions( 'adam', ...
        'MaxEpochs',100, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',learningrate, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',20, ...
        'LearnRateDropFactor',0.8, ...
        'L2Regularization',1e-3,...
        'Verbose',false, ...
        'ExecutionEnvironment',mydevice,...
        'Plots','training-progress');

%% 训练混合网络
% rng(0);
% 训练
net = trainNetwork(XrTrain,YrTrain,layers,options);
% 预测
YPredtrain = predict(net,XrTrain,"ExecutionEnvironment",mydevice,"MiniBatchSize",numFeatures);
% 结果
YPredtrain =double(YPredtrain');
% 反归�?��
CNNGRUoutput_train=method('reverse',YPredtrain,outputps);
CNNGRUoutput_train=double(CNNGRUoutput_train);
% 预测
YPredtest = predict(net,XrTest,"ExecutionEnvironment",mydevice,"MiniBatchSize",numFeatures);
% 结果
YPredtest =double(YPredtest');
% 反归�?��
CNNGRUoutput_test=method('reverse',YPredtest,outputps);
CNNGRUoutput_test=double(CNNGRUoutput_test);
%% 测试集误差评�?
CNNGRUerror_test=CNNGRUoutput_test'-output_test';
CNNGRUpererror_test=CNNGRUerror_test./output_test';
% RMSE
RMSEtest = sqrt(sumsqr(CNNGRUerror_test)/length(output_test));
% MAPE
MAPEtest = mean(abs(CNNGRUpererror_test));
disp("—�?—�?—�?CNNGRU网络模型测试数据—�?—�?—�?—�?—�?")
disp("    预测�?    真实�?    误差   ")
disp([CNNGRUoutput_test' output_test' CNNGRUerror_test])
%--------------------------------------------------------------------------
disp('CNNGRU测试平均绝对误差百分比MAPE');
disp(MAPEtest)
disp('CNNGRU测试均方根误差RMSE')
disp(RMSEtest)
%--------------------------------------------------------------------------
%% 数据可视�?
figure()
plot(CNNGRUoutput_test,'r-','linewidth',1)  
hold on
plot(output_test,'b-','linewidth',1)           
legend( '测试数据','实际数据','Location','NorthWest','FontName','华文宋体');
title('CNNGRU模型测试结果及真实�?','fontsize',12,'FontName','华文宋体')
xlabel('样本','fontsize',12,'FontName','华文宋体');
ylabel('数�?','fontsize',12,'FontName','华文宋体');
xlim([1 ntest]);
%-------------------------------------------------------------------------------------
figure()
plot(CNNGRUerror_test,'-','Color',[128 0 0]./255,'linewidth',1)   
legend('CNNGRU模型测试误差','Location','NorthEast','FontName','华文宋体')
title('CNNGRU模型测试误差','fontsize',12,'FontName','华文宋体')
ylabel('误差','fontsize',12,'FontName','华文宋体')
xlabel('样本','fontsize',12,'FontName','华文宋体')
xlim([1 ntest]);
%-------------------------------------------------------------------------------------
figure()
plot(CNNGRUpererror_test,'-','Color',[128 0 255]./255,'linewidth',1)   
legend('CNNGRU模型测试相对误差','Location','NorthEast','FontName','华文宋体')
title('CNNGRU模型测试相对误差','fontsize',12,'FontName','华文宋体')
ylabel('误差','fontsize',12,'FontName','华文宋体')
xlabel('样本','fontsize',12,'FontName','华文宋体')
xlim([1 ntest]);
