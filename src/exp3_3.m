% 实验3-3：探究信噪比对SFT和OMP算法频谱估计的l1误差的影响

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
M = 512;
A = randn(M, N);
% 生成IDFT矩阵
idft_mtx = dftmtx(N).' / N;

% 复振幅
a1 = 3 + 1j;
a2 = 2 - 2j;
% 数字频率
f1 = -0.15;
f2 = 0.25;

snr_values = linspace(10, 100, 20);

l1_error_sft = zeros(size(snr_values));
l1_error_omp = zeros(size(snr_values));

for idx = 1:length(snr_values)
    snr = snr_values(idx);
    % 复高斯噪声的标准差sigma
    % std_dev = sqrt((abs(a1) ^ 2 + abs(a2) ^ 2) / 10 ^ (snr / 10));
    std_dev = sqrt((abs(a1) ^ 2 + abs(a2) ^ 2) / snr);
    noise = std_dev / sqrt(2) * (randn(1, N) + 1j * randn(1, N));
    % 有噪声的频域系数信号x[n]
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
plot(snr_values, l1_error_sft, 'r-o');
hold on;
plot(snr_values, l1_error_omp, 'b-*');
title('信噪比对SFT和OMP算法频谱估计的l1误差的影响');
xlabel('信噪比 SNR');
ylabel('l1 error');
legend('SFT', 'OMP');
grid on;
saveas(gcf, './image/exp3_3.png');
