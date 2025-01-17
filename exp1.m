% 实验1：实现SFT算法

close all;
clear;
clc;

% 固定随机数种子
rng(2025);

% 信号长度
N = 2 ^ 12;
% 频谱稀疏度
K = 10;
% 分筐的个数B约为sqrt(NK)，整除N
B = 128;
% 循环次数L=O(logN)
L = ceil(log2(N));
% 定位循环用到的参数d<B/K
d = 4;
% 截断长度W<N
W = 400;

if mod(N, B) ~= 0
    error('error! B should divide N.');
end

% 信号频谱X[k]
X_k = zeros(1, N);

% 随机选择K个点赋非零值
nonzero_index = randperm(N, K);
% X_k(nonzero_index) = (0.5 + (1 - 0.5) * rand(size(nonzero_index))) .* exp(1j * 2 * pi * rand(size(nonzero_index)));

for m = nonzero_index
    % 模长为[0.5,1]内均匀分布
    magnitude = 0.5 + (1 - 0.5) * rand;
    % 辐角为[0,2*pi]内均匀分布
    phase = 2 * pi * rand;
    X_k(m) = magnitude * exp(1j * phase);
end

% 频谱X[k]做IDFT得时域信号x[n]
x_n = ifft(X_k, N);

X_est = sft(x_n, N, K, B, L, d, W);

% 绘制幅度谱真值
figure;
subplot(2, 1, 1);
plot((-N / 2:N / 2 - 1), abs(fftshift(X_k)));
title('频域幅度谱真值X[k]');
xlabel('频率/Hz');
ylabel('幅度');

subplot(2, 1, 2);
plot((-N / 2:N / 2 - 1), abs(fftshift(X_est)));
title('SFT 变换得到的X[k]');
xlabel('频率/Hz');
ylabel('幅度');
saveas(gcf, './image/exp1_spectrum.png');
