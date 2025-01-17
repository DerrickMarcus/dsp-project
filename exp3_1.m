% 实验3-1：实现OMP算法，与SFT算法对比

close all;
clear;
clc;

% 固定随机数种子
rng(2025);

% 信号长度
N = 2 ^ 12;
% 频谱稀疏度
K = 10;

X_k = zeros(1, N);
nonzero_index = randperm(N, K);

for m = nonzero_index
    % 模长为[0.5,1]内均匀分布
    magnitude = 0.5 + (1 - 0.5) * rand;
    % 辐角为[0,2*pi]内均匀分布
    phase = 2 * pi * rand;
    X_k(m) = magnitude * exp(1j * phase);
end

x_n = ifft(X_k, N);

%% SFT算法

% 分筐的个数B约为sqrt(NK)，整除N
B = 128;
% 循环次数L=O(logN)
L = ceil(log2(N));
% 定位循环用到的参数d<B/K
d = 4;
% 截断长度W<N
W = 400;

X_est_sft = sft(x_n, N, K, B, L, d, W);

%% OMP算法

% 观测维度M
M = 256;
% 生成高斯测量矩阵，得到观测信号y[n]
A = randn(M, N);
y_n = A * x_n.';
% 生成IDFT矩阵
idft_mtx = dftmtx(N).' / N;

X_est_omp = omp(y_n, A, idft_mtx, K);

%% 画图

figure;
subplot(3, 1, 1);
plot((-N / 2:N / 2 - 1), abs(fftshift(fft(x_n))));
title('频域幅度谱真值X[k]');
xlabel('频率/Hz');
ylabel('幅度');

subplot(3, 1, 2);
plot((-N / 2:N / 2 - 1), abs(fftshift(X_est_sft)));
title('SFT 变换得到的X[k]');
xlabel('频率/Hz');
ylabel('幅度');

subplot(3, 1, 3);
plot((-N / 2:N / 2 - 1), abs(fftshift(X_est_omp)));
title('OMP 算法得到的X[k]');
xlabel('频率/Hz');
ylabel('幅度');
saveas(gcf, './image/exp3_spectrum.png');
