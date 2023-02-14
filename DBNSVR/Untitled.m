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
        % 展开层
        sequenceUnfoldingLayer('Name','unfold')
        % 平滑层
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
% 批处理样本
MiniBatchSize =24;
% 最大迭代次数
MaxEpochs = 60;
% 学习率
learningrate = 0.005;
% 一些参数调整
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
