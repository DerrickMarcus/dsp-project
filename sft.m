% 稀疏傅里叶变换SFT

function X_hat = sft(x_n, N, K, B, L, d, W)
    % 参数:
    % x_n: 时域信号，大小[1,N]
    % N: 信号长度
    % K: 频谱稀疏度
    % B: 分筐的个数
    % L: 循环次数
    % d: 定位循环用到的参数
    % W: 窗函数的截断长度
    % 返回:
    % X_hat: 估计的频域信号，大小[1,N]

    if mod(N, B) ~= 0
        error('error! B should divide N.');
    end

    % 准备平坦窗函数g[n]
    rec_win = zeros(1, N);
    rec_win(1:W) = sinc(((0:W - 1) - W / 2) / B) / B;
    gauss_win = zeros(1, N);
    std_dev = B * log2(N); % 高斯窗的标准差
    gauss_win(1:W) = exp(- ((0:W - 1) - W / 2) .^ 2 / (2 * std_dev ^ 2));
    g_n = rec_win .* gauss_win;
    g_n = g_n ./ max(abs(g_n));
    G_k = fft(g_n);

    X_r = zeros(L, N); % 存储每次循环的估计值
    X_hat = zeros(1, N);

    for loop_cnt = 1:L
        % 生成随机参数sigma,tau
        % sigma为奇数，与N互素
        sigma = 2 * randi([0, N / 2 - 1]) + 1;
        tau = randi([1, N]);

        % 频谱随机重排
        p_n = x_n(mod(sigma * (0:N - 1) + tau, N) + 1); % 缩放平移
        y_n = p_n .* g_n; % 与窗函数时域相乘

        % 降采样FFT
        z_n = sum(reshape(y_n, B, []), 2);
        z_n = transpose(z_n);
        Z_k = fft(z_n, B);

        % 按幅度排序取出最大的d*K个
        [~, J] = sort(abs(Z_k), 'descend');
        J = J(1:d * K);

        % hash_sigma:[1,N]-->[1,B]
        hash_sigma = mod(round(mod(sigma * (0:N - 1), N) * B / N), B) + 1;
        % offset_sigma:[1,N]-->[1,N/2B],[N-N/2B-1,N]
        offset_sigma = mod(sigma * (0:N - 1) - round(sigma * (0:N - 1) * B / N) * N / B, N) + 1;

        % J的原像坐标集合I_r
        I_r = find(ismember(hash_sigma, J));
        % 估计幅度
        X_r(loop_cnt, I_r) = Z_k(hash_sigma(I_r)) .* exp(1j * 2 * pi * tau * (I_r - 1) / N) ./ G_k(offset_sigma(I_r));

    end

    % 找出出现次数大于L/2的频点坐标
    % nonzero_freq = find(sum(X_r ~= 0) > L / 2);

    % 找出出现次数最多的K个频点坐标
    [~, nonzero_freq] = sort(sum(X_r ~= 0), 'descend');
    nonzero_freq = nonzero_freq(1:K);

    for freq = nonzero_freq
        re = median(real(X_r(:, freq)));
        im = median(imag(X_r(:, freq)));
        X_hat(freq) = re + 1j * im;
    end

end
