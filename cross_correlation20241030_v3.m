clc;
clear;

num_v = 7;
num_h = 100;
N = num_v;
% P = string_to_bina;
P= randi([0 1], 2^num_v, 1);
num_sample=100;

original_cross_correlation = zeros(num_sample, num_h - 1);
% original_cross_correlation = zeros(num_sample, num_v - 1);%%%%求隐定显变的情况

p=P;
P = P / sum(P);
decimalNumbers = 0:2^N - 1;

basis = de2bi(decimalNumbers, N, 'left-msb');
dipole_basis = basis * 2 - 1;
X = dipole_basis';

mean_cross_correlation = zeros(1, num_h - 1);
std_cross_correlation = zeros(1, num_h - 1);
% mean_cross_correlation = zeros(1, num_v - 1);
% std_cross_correlation = zeros(1, num_v - 1);

% [W, A, B] = RBM_weight_train(num_v, num_h, P);

% 并行化外层循环 m=2:num_h

% parfor m = 2:num_h
%     for m = num_h:num_h
%         for l=1:20
            m=num_h;
            l=1;
             max_element = 1/(2^num_v);
             threshold = max_element / 2;
%             threshold = max_element / (2*l);
             threshold = threshold / l;
% parfor m = 2:num_v%%%%求隐定显变的情况
    % 临时变量，用于存储当前m值下的所有i迭代的结果
    temp_cross_correlation = zeros(num_sample, 1);
%     squaredSum=zeros(num_sample, 1);
    for i = 1:num_sample
        % 调用RBM_weight_train函数计算W, A, B
        [W, A, B] = RBM_weight_train(num_v, num_h, P);

        encrypt_alpha = A;

%%%%%%%%当计算显不变隐变时应注释掉这一块代码
        v_indices = randperm(7, 7); % 当前组合的索引
        
        v_original_part = A(v_indices(randperm(7))); % 选择的部分
        v_permutations = v_original_part;
        encrypt_alpha(v_indices) = v_permutations; % 替换为当前排列
% %%%%%%%%%

        indices = randperm(num_h, m);

        original_part = B(indices(randperm(m)));
          encrypt_beta = B;
        encrypt_beta(indices) = original_part;


%         [W,A,B]=RBM_weight_train(num_v,num_h,P);
% 
%         encrypt_beta = B;
%         indices = randperm(num_v, m); % 当前组合的索引
% 
%         %     original_part = A(indices); % 选择的部分
%         original_part = A(indices(randperm(m))); % 选择的部分
%         permutations = original_part;
%         encrypt_alpha=A;
%         encrypt_alpha(indices) = permutations;

        encrypt_beta1 = repmat(encrypt_beta, 1, 2^N);
        encrypt_W2 = W * X + encrypt_beta1;
        encrypt_alpha2 = encrypt_alpha' * X;

        encrypt_P_s = prod(cosh(encrypt_W2), 1) .* exp(encrypt_alpha2);
        encrypt_P_s = encrypt_P_s * 2^num_h;
        encrypt_Z = sum(encrypt_P_s);
        encrypt_P_s = encrypt_P_s / encrypt_Z;

%       max_element = max(encrypt_P_s(:));
%         max_element = 1/(2^num_v);
% %         threshold = max_element / 2;
%             threshold = max_element / 20;
        bipolar_encrypt_P_s = encrypt_P_s;
        bipolar_encrypt_P_s(encrypt_P_s > threshold) = 1;
        bipolar_encrypt_P_s(encrypt_P_s <= threshold) = 0;
        bipolar_encrypt_P_s = bipolar_encrypt_P_s';

%         product_values = bipolar_encrypt_P_s .* P;
%         product_values(i,:) = bipolar_encrypt_P_s' .* P';
%           product_values(i,:) = bipolar_encrypt_P_s' == p';
          product_values(i,:) = bipolar_encrypt_P_s';
%         temp_cross_correlation(i) = sum(product_values);
        
%         difference = p - bipolar_encrypt_P_s;
%         squaredSum(i) = sum(difference.^2);
    end

    % 计算并存储当前 m 值下的平均值和标准差
    mean_a=mean(product_values);
    std_a=std(product_values);
%     mean_cross_correlation(m - 1,l) = mean(temp_cross_correlation);
%     std_cross_correlation(m - 1,l) = std(temp_cross_correlation);
%         end
%     mean_squared(m-1)=mean(squaredSum);
%     std_squared(m-1)=std(squaredSum);
% end

% figure(1)
% % 绘制误差条图
% x = 1:1:20;
% % x = 2:1:num_v;
% errorbar(x, mean_cross_correlation, std_cross_correlation / 2);
% 
% mean_cross_correlation=mean_cross_correlation';
% std_cross_correlation=std_cross_correlation';

figure(1)
% 绘制误差条图
x = 1:1:128;
% x = 2:1:num_v;
mean_a=mean_a';
bar(mean_a);

figure(2)
bar(P);

% mean_a=0.8.*mean_a;
figure(3)
background=ones(2^num_v,1);
b1=bar(background, 'BarWidth', 1);
b1.FaceColor = [0.9529 0.9216 0.6980];

% 去掉边框
b1.EdgeColor = 'none';

hold on
b1=bar(p, 'BarWidth', 1);
b1.FaceColor = [0.7882 0.8902 0.8667];

% 去掉边框
b1.EdgeColor = 'none';

hold on
b1=bar(mean_a, 'BarWidth', 1);
b1.FaceColor = [0.9373 0.7451 0.7294];

% 去掉边框
b1.EdgeColor = 'none';
ylim([0 1]);
xlim([0.5 128.5]);
legend('0','1','single bit expectation');
% mean_cross_correlation=mean_cross_correlation';
% std_cross_correlation=std_cross_correlation';

% normal_mean_squared =mean_squared / (size(P,1)*size(P,2));
% normal_std_squared =std_squared / (size(P,1)*size(P,2));
% 
% figure(2)
% errorbar(x, normal_mean_squared, normal_std_squared / 2);
% 归一化操作
% normal_mean_cross_correlation = mean_cross_correlation / nnz(P);
% normal_std_cross_correlation = std_cross_correlation / nnz(P);
% normal_mean_cross_correlation = normal_mean_cross_correlation';
% normal_std_cross_correlation = normal_std_cross_correlation';
