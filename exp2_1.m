% 实验2-1：SFT算法运行时间与K的关系

close all;
clear;
clc;

% 固定随机数种子
rng(2025);

% 信号长度
N = 2 ^ 12;
% 分筐的个数B约为sqrt(NK)，整除N
B = 128;
% 循环次数L=O(logN)
L = ceil(log2(N));
% 定位循环用到的参数d<B/K
d = 2;
% 截断长度W<N
W = 400;

% 频谱稀疏度
K_values = 2:floor(B / d);
run_times = zeros(size(K_values));

if mod(N, B) ~= 0
    error('error! B should divide N.');
end

for k_idx = 1:length(K_values)
    K = K_values(k_idx);

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

    tic;
    X_est = sft(x_n, N, K, B, L, d, W);
    run_times(k_idx) = toc;
    disp(['K = ', num2str(K), ', run time:', num2str(run_times(k_idx)), ' s']);
end

figure;
plot(sqrt(K_values), run_times, '-o');
title('运行时间与稀疏度 K 的关系曲线');
xlabel('sqrt(K)');
ylabel('运行时间 /s');
grid on;
saveas(gcf, './image/run_time_vs_K.png');
