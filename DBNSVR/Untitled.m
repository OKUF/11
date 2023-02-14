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
