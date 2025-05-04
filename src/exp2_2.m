% 实验2-2：SFT算法运行时间与N的关系

close all;
clear;
clc;

% 固定随机数种子
rng(2025);

% 信号长度
N_values = 2 .^ (10:20);
run_times = zeros(size(N_values));

% 频谱稀疏度
K = 10;
% 分筐的个数B约为sqrt(NK)，整除N
B = 128;
% 循环次数L=O(logN)
L = 15;
% 定位循环用到的参数d<B/K
d = 2;
% 截断长度W<N
W = 400;

for n_idx = 1:length(N_values)
    N = N_values(n_idx);

    if mod(N, B) ~= 0
        error('error! B should divide N.');
    end

    % 信号频谱X[k]
    X_k = zeros(1, N);

    % 随机选择K个点赋非零值
    nonzero_index = randperm(N, K);

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
    run_times(n_idx) = toc;
    disp(['N = ', num2str(N), ', run time:', num2str(run_times(n_idx)), ' s']);

end

figure;
plot(log2(N_values), run_times, '-o');
xlabel('信号长度 N');
ylabel('运行时间 /s');
title('运行时间与信号长度 N 的关系曲线');
grid on;
saveas(gcf, './image/run_time_vs_N1.png');

figure;
plot((log2(N_values)) .^ (3/2) .* sqrt(N_values), run_times, '-o');
xlabel('log^{1.5}(N) * sqrt(N)');
ylabel('运行时间 /s');
title('运行时间与信号长度 N 的关系曲线');
grid on;
saveas(gcf, './image/run_time_vs_N2.png');
