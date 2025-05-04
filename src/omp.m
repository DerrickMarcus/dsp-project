% 压缩感知OMP算法

function x_hat = omp(y, measure_mtx, trans_mtx, sparsity)
    % 参数:
    % y: 观测信号，大小[M,1]
    % measure_mtx: 测量矩阵，大小[M,N]
    % trans_mtx: 变换矩阵，大小[N,N]
    % sparsity: 稀疏度
    % 返回:
    % X_hat: 估计信号，大小[N,1]

    sensing_mtx = measure_mtx * trans_mtx; % 感知矩阵
    [~, N] = size(sensing_mtx);
    x_hat = zeros(N, 1);
    residual = y; % 初始化残差r=y
    index = []; % 初始化索引集合S
    threshold = 1e-6; % 设置阈值

    for cnt = 1:sparsity
        correlations = abs(sensing_mtx.' * residual);
        correlations(index) = -inf;
        [~, i_k] = max(correlations);
        index = [index, i_k];
        x_hat(index) = pinv(sensing_mtx(:, index)) * y;
        residual = y - sensing_mtx * x_hat;

        if norm(residual) < threshold
            break;
        end

    end

end
