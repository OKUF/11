%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('text.xlsx');

%%  数据分析
num_size = 0.7;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  模型预训练
dbn.sizes      = [50, 50];           % 隐藏层节点数
opts.numepochs = 500;                % 训练次数
opts.batchsize = 12;                 % 每次训练样本个数 需满足：（M / batchsize = 整数）
opts.momentum  = 0;                  % 动量参数
opts.alpha     = 0.1;                % 学习率

dbn = dbnsetup(dbn, p_train, opts);  % 建立模型
dbn = dbntrain(dbn, p_train, opts);  % 训练模型

%%  训练权重移植，添加输出层
nn = dbnunfoldtonn(dbn, outdim);

%%  反向调整网络
opts.numepochs = 3000;                            % 反向微调次数
opts.batchsize = 12;                              % 每次反向微调样本数 需满足：（M / batchsize = 整数）

nn.activation_function = 'sigm';                  % 激活函数
nn.learningRate = 2;                              % 学习率
nn.momentum = 0.5;                                % 动量参数
nn.scaling_learningRate = 1;                      % 学习率的比例因子

[nn, loss] = nntrain(nn, p_train, t_train, opts); % 反向微调训练

%%  模型预测
t_sim1 = nnpredict(nn, p_train);
t_sim2 = nnpredict(nn, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  绘制损失函数曲线
figure
plot(1 : length(loss), loss, 'b-', 'LineWidth', 1)
xlim([1, length(loss)])
xlabel('迭代次数')
ylabel('误差损失')
legend('损失函数')
title('损失函数')
grid

%%  绘图
figure
plot(1: M, T_train, '-s', 1: M, T_sim1, '-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, '-s', 1: N, T_sim2, '-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比';['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2')^2 / norm(T_test -  mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])


