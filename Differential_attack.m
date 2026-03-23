clc;
clear;

num_v = 7;
num_h = 50;
N = num_v; %%% 第一个玻尔兹曼机的比特数
n = 10000; %%% 迭代次数
alp = 25; %%% 学习率
decay_rate = 0.9999; % 学习率衰减
jiezhi = 10^(-3); %%% 迭代停止条件
num_hamming=100;%%%计算汉明距离时用的时长

P= randi([0 1], 2^num_v, 1);

% P = rand(2^N, 1) * 0.1;
% tuxiang = [3, 26, 35, 54];
% P(tuxiang) = P(tuxiang) + 1;

p=P;
P = P / sum(P);

%%% 将像素值大小转换为概率值，并按照二进制编码放入对应的矩阵位置中
E = -log(P);
decimalNumbers = 0:2^N-1;

% 初始化二进制数组并转换为 dipole_basis
basis = de2bi(decimalNumbers, N, 'left-msb');
dipole_basis = basis * 2 - 1; % 将 0 转换为 -1

X = dipole_basis';

%%% 现在已经准备好数据了

%%%计算平均KL引入的循环
% KL=zeros(100,1);
% for l=1:100


%%% 建立相应的权重矩阵，确定未知参量的位置，写出对应的损失函数，设定截止步长
W = rand(num_h, num_v) * 2 - 1;
alpha = rand(num_v, 1) * 2 - 1;
beta = rand(num_h, 1) * 2 - 1;



%%%循环迭代过程



for i=1:n


tic;

    % 假设 beta, W, X 和 alpha 已经定义
    % 优化这段代码
    beta1 = repmat(beta, 1, 2^N);
    W2 = W * X + beta1;
    alpha2 = alpha' * X;





% 假设 W2 和 alpha2 已经定义
P_s = prod(cosh(W2), 1) .* exp(alpha2); % 逐列计算 cosh 乘积并乘以 exp(alpha2)
P_s = P_s * 2^num_h; % 乘以 2^num_h
Z = sum(P_s); % 计算 Z
P_s = P_s / Z; % 归一化 P_s


    record_i=n;

    W1=W;
    alpha1=alpha;
    beta1=beta;




% 假设 W2, P_s, P, X, 和 alp 已经定义
W_tidu = zeros(num_h, num_v);
for j = 1:num_h %%% 更新每个 W 分量
    for k = 1:num_v
        A = -sum(P_s .* tanh(W2(j, :)) .* X(k, :));
        W_tidu(j, k) = sum(P' .* tanh(W2(j, :)) .* X(k, :)) + A;
    end
end
W1 = W + alp * W_tidu / 2^N;


% 假设 P_s, P, X, 和 alp 已经定义
alpha_tidu = zeros(num_v, 1);

% 矢量化计算 A
A = -sum(P_s .* X, 2);

% 矢量化计算 alpha_tidu
alpha_tidu = sum(P' .* X, 2) + A;

% 更新 alpha1
alpha1 = alpha + alp * alpha_tidu / 2^N;




% 假设 P_s, P, W2, 和 alp 已经定义
beta_tidu = zeros(num_h, 1);

% 矢量化计算 A
A = -sum(P_s .* tanh(W2), 2);

% 矢量化计算 beta_tidu
beta_tidu = sum(P' .* tanh(W2), 2) + A;

% 更新 beta1
beta1 = beta + alp * beta_tidu / 2^N;





    alpha=alpha1;
    beta=beta1;
    W=W1;



% tic;
    beta1 = repmat(beta, 1, 2^N);
    W2 = W * X + beta1;
    alpha2 = alpha' * X;



P_s = prod(cosh(W2), 1) .* exp(alpha2); % 逐列计算 cosh 乘积并乘以 exp(alpha2)
P_s = P_s * 2^num_h; % 乘以 2^num_h
Z = sum(P_s); % 计算 Z
P_s = P_s / Z; % 归一化 P_s

% time1 = toc;
    %%%计算KL散度



% 假设 P 和 P_s 已经定义
non_zero_indices = P' ~= 0;
record_KL(i) = sum(P(non_zero_indices)' .* log(P(non_zero_indices)' ./ P_s(non_zero_indices)));



    %%%计算更新用到的梯度的大小
    W_tidu=reshape(W_tidu,num_h*num_v,1);
    alpha_tidu=reshape(alpha_tidu,num_v,1);
    beta_tidu=reshape(beta_tidu,num_h,1);
    tidu=[W_tidu;alpha_tidu;beta_tidu];
    record_norm(i)=norm(tidu);

    % 计算梯度的范数
grad_norm = norm(tidu);

% 检查是否达到了停止条件
if grad_norm <= jiezhi
    record_i = i;
%     disp(i); % 使用disp代替display，disp是MATLAB中更常用的输出函数
    break;
end

% 调整学习率
if i > 1
    if record_KL(i) == record_KL(i-1)
%   if ((record_KL(i)-record_KL(i-1))<0)&(abs(record_KL(i)-record_KL(i-1))<10^(-4)) 
%         alp = alp + 0.1;
          alp = alp + 0.1*alp;
    elseif record_KL(i) > record_KL(i-1)
%         alp = max(alp - 0.01, 0); % 确保alp不会低于0
          alp = max(alp - 0.01*alp, 0); % 确保alp不会低于0
        if alp <= 0
            record_i = i;
            break;
        end
    end
end

time1 = toc;
% alp=alp*decay_rate; 

end
% 
record_KL=record_KL';


alpha_binary=alpha>0;
beta_binary=beta>0;


Hamming_distance_alpha=zeros(num_hamming,2^num_v+1);
Hamming_distance_beta=zeros(num_hamming,2^num_v+1);


alpha_matrix=zeros(num_v,num_hamming,2^num_v+1);
beta_matrix=zeros(num_h,num_hamming,2^num_v+1);


for l=1:1:2^num_v+1
    
% Hamming_distance_alpha=zeros(num_hamming,1);
% Hamming_distance_beta=zeros(num_hamming,1);

% 1;
parfor m=1:num_hamming

alp = 25;
numbers = randperm(128, l-1); 
P=p;
P(numbers)=1-P(numbers);


P = P / sum(P);

%%% 将像素值大小转换为概率值，并按照二进制编码放入对应的矩阵位置中
% E = -log(P);
decimalNumbers = 0:2^N-1;

% 初始化二进制数组并转换为 dipole_basis
basis = de2bi(decimalNumbers, N, 'left-msb');
dipole_basis = basis * 2 - 1; % 将 0 转换为 -1

X = dipole_basis';

%%% 现在已经准备好数据了

%%%计算平均KL引入的循环
% KL=zeros(100,1);
% for l=1:100


%%% 建立相应的权重矩阵，确定未知参量的位置，写出对应的损失函数，设定截止步长
W = rand(num_h, num_v) * 2 - 1;
alpha = rand(num_v, 1) * 2 - 1;
beta = rand(num_h, 1) * 2 - 1;

record_KL_local = zeros(n,1);
record_norm_local = zeros(n,1);





for i=1:n


% tic;

    % 假设 beta, W, X 和 alpha 已经定义
    % 优化这段代码
    beta1 = repmat(beta, 1, 2^N);
    W2 = W * X + beta1;
    alpha2 = alpha' * X;





% 假设 W2 和 alpha2 已经定义
P_s = prod(cosh(W2), 1) .* exp(alpha2); % 逐列计算 cosh 乘积并乘以 exp(alpha2)
P_s = P_s * 2^num_h; % 乘以 2^num_h
Z = sum(P_s); % 计算 Z
P_s = P_s / Z; % 归一化 P_s


    record_i=n;

    W1=W;
    alpha1=alpha;
    beta1=beta;




% 假设 W2, P_s, P, X, 和 alp 已经定义
W_tidu = zeros(num_h, num_v);
for j = 1:num_h %%% 更新每个 W 分量
    for k = 1:num_v
        A = -sum(P_s .* tanh(W2(j, :)) .* X(k, :));
        W_tidu(j, k) = sum(P' .* tanh(W2(j, :)) .* X(k, :)) + A;
    end
end
W1 = W + alp * W_tidu / 2^N;


% 假设 P_s, P, X, 和 alp 已经定义
alpha_tidu = zeros(num_v, 1);

% 矢量化计算 A
A = -sum(P_s .* X, 2);

% 矢量化计算 alpha_tidu
alpha_tidu = sum(P' .* X, 2) + A;

% 更新 alpha1
alpha1 = alpha + alp * alpha_tidu / 2^N;




% 假设 P_s, P, W2, 和 alp 已经定义
beta_tidu = zeros(num_h, 1);

% 矢量化计算 A
A = -sum(P_s .* tanh(W2), 2);

% 矢量化计算 beta_tidu
beta_tidu = sum(P' .* tanh(W2), 2) + A;

% 更新 beta1
beta1 = beta + alp * beta_tidu / 2^N;





    alpha=alpha1;
    beta=beta1;
    W=W1;



% tic;
    beta1 = repmat(beta, 1, 2^N);
    W2 = W * X + beta1;
    alpha2 = alpha' * X;



P_s = prod(cosh(W2), 1) .* exp(alpha2); % 逐列计算 cosh 乘积并乘以 exp(alpha2)
P_s = P_s * 2^num_h; % 乘以 2^num_h
Z = sum(P_s); % 计算 Z
P_s = P_s / Z; % 归一化 P_s

% time1 = toc;
    %%%计算KL散度



% 假设 P 和 P_s 已经定义
non_zero_indices = P' ~= 0;
% record_KL(i) = sum(P(non_zero_indices)' .* log(P(non_zero_indices)' ./ P_s(non_zero_indices)));
record_KL_local(i) = sum(P(non_zero_indices)' .* log(P(non_zero_indices)' ./ P_s(non_zero_indices)));


    %%%计算更新用到的梯度的大小
    W_tidu=reshape(W_tidu,num_h*num_v,1);
    alpha_tidu=reshape(alpha_tidu,num_v,1);
    beta_tidu=reshape(beta_tidu,num_h,1);
    tidu=[W_tidu;alpha_tidu;beta_tidu];
%     record_norm(i)=norm(tidu);

    % 计算梯度的范数
grad_norm = norm(tidu);


% record_KL_local(i) = sum( ... );
record_norm_local(i) = norm(tidu);

% 检查是否达到了停止条件
if grad_norm <= jiezhi
    record_i = i;
%     disp(i); % 使用disp代替display，disp是MATLAB中更常用的输出函数
    break;
end

% 调整学习率
% if i > 1
%     if record_KL(i) == record_KL(i-1)
% %   if ((record_KL(i)-record_KL(i-1))<0)&(abs(record_KL(i)-record_KL(i-1))<10^(-4)) 
% %         alp = alp + 0.1;
%           alp = alp + 0.1*alp;
%     elseif record_KL(i) > record_KL(i-1)
% %         alp = max(alp - 0.01, 0); % 确保alp不会低于0
%           alp = max(alp - 0.01*alp, 0); % 确保alp不会低于0
%         if alp <= 0
%             record_i = i;
%             break;
%         end
%     end
% end


if i > 1
    if record_KL_local(i) == record_KL_local(i-1)
%   if ((record_KL(i)-record_KL(i-1))<0)&(abs(record_KL(i)-record_KL(i-1))<10^(-4)) 
%         alp = alp + 0.1;
          alp = alp + 0.1*alp;
    elseif record_KL_local(i) > record_KL_local(i-1)
%         alp = max(alp - 0.01, 0); % 确保alp不会低于0
          alp = max(alp - 0.01*alp, 0); % 确保alp不会低于0
        if alp <= 0
            record_i = i;
            break;
        end
    end
end


% time1 = toc;
% alp=alp*decay_rate; 

end

alpha_binary_1=alpha>0;
beta_binary_1=beta>0;

alpha_matrix(:,m,l)=alpha_binary_1;
beta_matrix(:,m,l)=beta_binary_1;

% diff_matrix_alpha = alpha_binary_1 ~= alpha_binary;          % 找出不同的位置（返回逻辑矩阵）
% Hamming_distance_alpha(m,l)= sum(diff_matrix_alpha(:));

Hamming_distance_alpha(m,l)= sum(alpha_binary_1.*alpha_binary);


% diff_matrix_beta = beta_binary_1 ~= beta_binary;          % 找出不同的位置（返回逻辑矩阵）
% Hamming_distance_beta(m,l)= sum(diff_matrix_beta(:));
Hamming_distance_beta(m,l)= sum(beta_binary_1.*beta_binary);


    end

% Hamming_distance_alpha_ave(l)=sum(Hamming_distance_alpha)/num_hamming;
% Hamming_distance_beta_ave(l)=sum(Hamming_distance_beta)/num_hamming;


end


% Hamming_distance_alpha_ave=sum(Hamming_distance_alpha)/(num_hamming*sum(alpha_binary));
% Hamming_distance_beta_ave=sum(Hamming_distance_beta)/(num_hamming*sum(beta_binary));



frac_hamming_alpha=Hamming_distance_alpha./sum(alpha_binary);
frac_hamming_beta=Hamming_distance_beta./sum(beta_binary);

ave_frac_hamming_alpha=mean(frac_hamming_alpha);
std_frac_hamming_alpha=std(frac_hamming_alpha);

ave_frac_hamming_beta=mean(frac_hamming_beta);
std_frac_hamming_beta=std(frac_hamming_beta);

ave_frac_hamming_alpha=ave_frac_hamming_alpha';
std_frac_hamming_alpha=std_frac_hamming_alpha';
ave_frac_hamming_beta=ave_frac_hamming_beta';
std_frac_hamming_beta=std_frac_hamming_beta';


% figure(1)
% plot(0:128,Hamming_distance_alpha_ave);
% xlabel('$\Delta P$', 'Interpreter', 'latex', 'FontSize', 14);
% ylabel('$\Delta \mathbf{\alpha}$', 'Interpreter', 'latex', 'FontSize', 14);
% title('n=7');
% ylim([0,1]);
% xlim([0,128]);
% 
% figure(2)
% plot(0:128,Hamming_distance_beta_ave);
% xlabel('$\Delta P$', 'Interpreter', 'latex', 'FontSize', 14);
% ylabel('$\Delta \mathbf{\beta}$', 'Interpreter', 'latex', 'FontSize', 14);
% title('m=20');
% ylim([0,1]);
% xlim([0,128]);

figure(3);
errorbar(0:128, ave_frac_hamming_alpha, std_frac_hamming_alpha./2, 'o-');
xlabel('$\Delta P$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('cross correlation($\Delta \mathbf{\alpha}$)', 'Interpreter', 'latex', 'FontSize', 14);
title('n=7');
ylim([0,1]);
xlim([0,128]);
grid on;

figure(4);
errorbar(0:128, ave_frac_hamming_beta, std_frac_hamming_beta./2, 'o-');
xlabel('$\Delta P$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('cross correlation($\Delta \mathbf{\beta}$)', 'Interpreter', 'latex', 'FontSize', 14);
title('m=50');
ylim([0,1]);
xlim([0,128]);
grid on;


%%%%保存文件
save('binary_matrix_1.mat','alpha_binary','beta_binary','alpha_matrix','beta_matrix');

% KL(l)=record_KL(end);
% 
% end
%%%%%%%%%%%计算打乱偏置后的分布

% encrypt_num_v=randperm(num_v);
% 
% encrypt_num_h=randperm(num_h);
% encrypt_num_v=1:num_v;
% encrypt_num_h=1:num_h;
% encrypt_num_v(5)=3;
% encrypt_num_v(3)=5;

% for i=1:num_v
%     encrypt_alpha(i,1)=alpha(encrypt_num_v(i));
% end
% 
% for i=1:num_h
%     encrypt_beta(i,1)=beta(encrypt_num_h(i));
% end
% 
% % encrypt_beta=encrypt_beta';
% % encrypt_alpha= encrypt_alpha';
% 
% encrypt_beta1 = repmat(encrypt_beta, 1, 2^N);
% encrypt_W2 = W * X + encrypt_beta1;
% encrypt_alpha2 = encrypt_alpha' * X;
% 
% 
% 
% 
% % 假设 W2 和 alpha2 已经定义
% encrypt_P_s = prod(cosh(encrypt_W2), 1) .* exp(encrypt_alpha2); % 逐列计算 cosh 乘积并乘以 exp(alpha2)
% encrypt_P_s = encrypt_P_s * 2^num_h; % 乘以 2^num_h
% encrypt_Z = sum(encrypt_P_s); % 计算 Z
% encrypt_P_s = encrypt_P_s / encrypt_Z; % 归一化 P_s


%%%%%%%%%%%%

% quanzhong=[zeros(num_v,num_v),W'];
% quanzhong=[quanzhong;zeros(num_h,num_h+num_v)];
% quanzhong=quanzhong+quanzhong';
% quanzhong=quanzhong;
% 
% pianzhi=[alpha;beta];
% % pianzhi=[encrypt_alpha;beta];
% pianzhi=pianzhi;

%%%画图
% figure(1)
% record_i=1:record_i;
% plot(record_i,record_norm);
% 
% figure(2)
% plot(1:record_i,record_KL,'b','LineWidth',3);
% xlabel('iteration','FontSize', 15);
% ylabel('KL divergence','FontSize', 15);
% % set(gca, 'XScale', 'log') 
% 
% record_norm=record_norm';
% record_i=record_i';
% 
% % pianzhi=pianzhi';
% 
% E=E';
% record_KL=record_KL';


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %构建两个玻尔兹曼机，让其随机波动，同时在每个玻尔兹曼机每一次迭代完所有的p-bit后，将两个玻尔兹曼机的结果作加和，最后统计和的概率分布



% n2_bit=num_v+num_h;
% 
% 
% J2=quanzhong;
% 
% 
% H2=pianzhi;
% 
% pbit2=ones(n2_bit,1);
% 
% I20=1;%%%玻尔兹曼机的逆温度
% 
% N1=1000;%%%采样次数
% N_par=1;
% % if isempty(gcp('nocreate'))
% %     parpool;
% % end
% % parfor k = 1:N_par
% %  for k = 1:N_par
% % 使用矢量化操作初始化 pbit2
% pbit2 = sign(randi([0, 1], n2_bit, 1) * 2 - 1.5);
% 
% 
% 
% 
% 
% %%%采样过程
% 
% % 假设 n2_bit, N1, J2, H2, I20, 和 pbit2 已经定义
% pbit2_record = zeros(n2_bit, N1);
% I = zeros(n2_bit, 1); % 预分配 I 数组
% m = zeros(n2_bit, 1); % 预分配 m 数组
% 
% tic;
% for i = 1:N1
%     % 记录每次采样 p 比特的状态
%     pbit2_record(:, i) = pbit2;
%     
%     % 随机更新顺序
%     updateorder_2 = randperm(n2_bit);
%    
%     for j = 1:n2_bit
%         % 计算输入某 p-bit 的电流
%         idx = updateorder_2(j);
%          
%         I(idx) = I20 * (J2(idx, :) * pbit2 + H2(idx));
%         
%         % 根据输入电流进行翻转
%         m(idx) = sign((rand * 2 - 1) + tanh(I(idx)));
%         
%         pbit2(idx) = m(idx);
%     end
% 
% end
% time2 = toc;
% 
% 
% pbit2_record=pbit2_record(1:num_v,:);
% 
% % 将 -1 和 1 转换为 0 和 1
% bit2 = (pbit2_record == 1); % 逻辑索引，等于 1 的位置变为 1，其余为 0
% 
% 
% % 将样本转换为对应的十进制
% % deci2(k,:) = bi2de(bit2', 'left-msb');
% deci2= bi2de(bit2', 'left-msb');
% 
% 
% % end
% deci2_flat = deci2(:);
% % 统计某一 m 配置出现的次数
% num2 = histcounts(deci2_flat, 0:2^num_v);
% 
% %%%统计某一m配置出现的概率
% num2=num2./(N1*N_par);%%%统计值




% 第二个玻尔兹曼机统计与理论结果
% figure(3)
% 
% hengzuobiao_2=1:2^(num_v);
% % plot(hengzuobiao_2,p_2,'r.');
% plot(hengzuobiao_2,P_s,'.b');%%%理论值
% hold on
% % plot(hengzuobiao_2,num2,'ro');
% % plot(hengzuobiao_2,num2,'.r');%%%统计值
% % hold on
% % plot(hengzuobiao_2,P,'*');
% plot(hengzuobiao_2,P,'k');%%%样本
% hold on
% % plot(hengzuobiao_2,encrypt_P_s,'.g');%%%保密处理后的
% % hold on
% xlabel('x');
% ylabel('P');
% title('learning rate');
% legend('analysis','statistics','sample');

% 
% P_s=P_s';%%%解析值
% img_P_s=reshape(P_s,32,32)';
% img_P_s=reshape(P_s,16,8)';
% figure(4)
% imagesc(img_P_s) % 设置CLim属性
% colormap(flip(gray));
% colorbar;
% title('reconstruction-plaintext', 'FontSize', 18);
% 
% num2=num2';%%%统计值
% % img_num2=reshape(num2,32,32)';
% img_num2=reshape(num2,16,8)';
% figure(5)
% imagesc(img_num2) % 设置CLim属性
% colormap(flip(gray));
% colorbar;
% title('RBM-plaintext','FontSize', 18);
% 
% % img_P=reshape(P,32,32)';%%%原始图像
% img_P=reshape(P,16,8)';%%%原始图像
% figure(6)
% imagesc(img_P) % 设置CLim属性
% colormap(flip(gray));
% colorbar;
% title('plaintext', 'FontSize', 18);
% %
% % 去掉横纵坐标轴的刻度
% xticks([]);
% yticks([]);
% 
% % 添加网格线
% grid on
% 
% 
% encrypt_P_s=encrypt_P_s';%%%保密处理后
% img_encrypt_P_s=reshape(encrypt_P_s,32,32)';
% img_encrypt_P_s=reshape(encrypt_P_s,16,8)';
% figure(7)
% imagesc(img_encrypt_P_s) % 设置CLim属性
% colormap(flip(gray));
% colorbar;
% title('encryption', 'FontSize', 18);
% 
% cit=[W',alpha];
% % cit=[W',encrypt_alpha];
% cit_1=[beta',0];
% cit=[cit;cit_1];
% % cit=[quanzhong pianzhi];
% figure(8)
% imagesc(cit);
% colormap("parula");
% % colormap(flip(gray));
% colorbar;
% title('ciphertext', 'FontSize', 18);
% axis equal;
% set(gca, 'YLim', [1 size(cit, 1)]); % 调整 Y 轴的显示范围
% 
% axis_x=zeros(808,1);
% for i=1:8
%     for j=1:101
%         axis_x((i-1)*101+j)=i;
%     end
% end
%  
% axis_y1=1:101;
% axis_y1=axis_y1';
% axis_y=axis_y1;
% for i=1:7
% axis_y=[axis_y;axis_y1];
% end
% 
% axis_ciphertext=[axis_x axis_y ciphertext];





