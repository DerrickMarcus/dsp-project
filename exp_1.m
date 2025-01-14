close;
clear all;
clc;

% 参数设置
N = 1024; % 信号长度
k = 10; % 频域稀疏点个数

% 初始化频域信号S[i]
S = zeros(1, N);

% 随机选择k个点的索引
indices = randperm(N, k);

% 为选中的点赋值
for m = 1:k
    % 模长为0.5到1之间均匀分布的随机值
    magnitude = 0.5 + (1 - 0.5) * rand;
    % 辐角为0到2*pi之间均匀分布的随机值
    angle = 2 * pi * rand;
    % 构造复数
    S(indices(m)) = magnitude * exp(1j * angle);
end

% 对频域信号S[i]做IDFT得到时域信号x[n]
x = ifft(S);

% 绘制时域信号
figure;
plot(real(x));
title('时域信号x[n]');
xlabel('n');
ylabel('实部');

% 绘制频域信号的幅度谱
figure;
plot(abs(fftshift(fft(x))));
title('频域信号幅度谱');
xlabel('频率');
ylabel('幅度');