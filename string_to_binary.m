clc;
clear;
% 示例字符串
% inputStr = 'Data security ensures privacy. Encryption methods prevent
% unauthorized access to confidential information and protect user
% data.';%学习率为200
inputStr = 'Fudan University';
% inputStr = 'íº×ä©¹ÝR"j¿OB}';
% 统计字符串的字符数
numChars = strlength(inputStr);

% 调用函数
for i=1:numChars
binaryResult(i,:) = stringToBinary(inputStr(i));
end
binaryResult=binaryResult';
plt=binaryResult(:,1);
for i=2:numChars
plt=[plt;binaryResult(:,i)];
end
% plt=reshape(binaryResult,1,numChars*8);
% 显示结果
% disp('Binary result:');
% disp(binaryResult);
% 
% % 将binaryResult转换成数字
% numericBinaryResult = binaryResult - '0'; % 将字符'0'和'1'转换成数字0和1
% 
% % 确定矩阵的大小
% numElements = length(numericBinaryResult);
% numRows = numElements; % 假设矩阵有16行
% numCols = numElements / numRows;
% 
% % 将数字数组转换为二维矩阵
% binaryMatrix = reshape(numericBinaryResult, [numRows, numCols]);

% % 显示结果
% disp('Binary matrix:');
% disp(binaryMatrix);

