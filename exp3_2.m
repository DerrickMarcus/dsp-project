% 实验3-2：探究信号两个分量的频率间隔对SFT和OMP算法频谱估计的l1误差的影响

close all;
clear;
clc;

% 固定随机数种子
rng(2025);

% 信号长度
N = 2 ^ 12;
% 频谱稀疏度
K = 2;
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

% 观测维度M
M = 256;
A = randn(M, N);
% 生成IDFT矩阵
idft_mtx = dftmtx(N).' / N;

% 复振幅
a1 = 3 + 1j;
a2 = 2 - 2j;
% 数字频率
f1 = -0.5;
delta_f = linspace(0, 1, 10);
delta_f = delta_f(2:end - 1);

% 复高斯噪声的标准差sigma
std_dev = 0.1;
% noise = std_dev / sqrt(2) * (randn(1, N) + 1j * randn(1, N));

l1_error_sft = zeros(size(delta_f));
l1_error_omp = zeros(size(delta_f));

for idx = 1:length(delta_f)
    f2 = f1 + delta_f(idx);

    noise = std_dev / sqrt(2) * (randn(1, N) + 1j * randn(1, N));
    % 有噪声的频域稀疏信号x[n]
    x_n = a1 * exp(1j * 2 * pi * f1 * (0:N - 1)) + ...
        a2 * exp(1j * 2 * pi * f2 * (0:N - 1)) + noise;

    X_k = fft(x_n);

    % SFT算法
    X_est_sft = sft(x_n, N, K, B, L, d, W);
    % X_est_sft = X_est_sft / max(abs(X_est_sft)) * max(abs(X_k));
    X_est_sft = X_est_sft / sum(abs(X_est_sft)) * sum(abs(X_k));
    l1_error_sft(idx) = sum(abs(X_est_sft - X_k)) / K;

    % OMP算法
    X_est_omp = omp(A * x_n.', A, idft_mtx, K).';
    % X_est_omp = X_est_omp / max(abs(X_est_omp)) * max(abs(X_k));
    X_est_omp = X_est_omp / sum(abs(X_est_omp)) * sum(abs(X_k));
    l1_error_omp(idx) = sum(abs(X_est_omp - X_k)) / K;
end

figure;
plot(delta_f, l1_error_sft, 'r-o');
hold on;
plot(delta_f, l1_error_omp, 'b-*');
title('频率间隔对SFT和OMP算法频谱估计的l1误差的影响');
xlabel('数字频率间隔 \Delta f');
ylabel('l1 error');
legend('SFT', 'OMP');
grid on;
saveas(gcf, './image/exp3_2.png');
