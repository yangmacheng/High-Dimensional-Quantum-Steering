function max_value = omegak_batching_optimized_Bloch(k, blochvector, d) 
% OMEGAK_BATCHING_OPTIMIZED_BLOCH 计算Bloch参数化下k个向量投影和的最大谱范数
%
% 输入:
%   k:           选取的向量数量 (标量)
%   blochvector: (d^2-1) x num 的矩阵，每一列代表一个可观测量的 Bloch 矢量参数
%   d:           希尔伯特空间维度 (支持 d=2 或 d=3)
%
% 输出:
%   max_value:   所有可能的 k 个特征向量组合中，投影和矩阵的最大谱范数
%
% 算法逻辑:
%   1. 根据 d 生成 Pauli (d=2) 或 Gell-Mann (d=3) 基矩阵。
%   2. 将 Bloch 矢量还原为测量，并对角化得到特征向量(测量基)。
%   3. 预计算所有特征向量的投影矩阵 P = |v><v|。
%   4. 使用组合数反解算法并行搜索最大谱范数。

    % ================= 1. 基础参数与生成元准备 =================
    [dim_bloch, num_observables] = size(blochvector);
    
    % 检查维度一致性
    expected_dim = d^2 - 1;
    if dim_bloch ~= expected_dim
        error('Bloch矢量维度 (%d) 与希尔伯特空间维度 d=%d (需 %d 维) 不匹配', dim_bloch, d, expected_dim);
    end

    % 预分配生成元矩阵 (SU(2) 或 SU(3))
    basis_matrices = zeros(d, d, expected_dim);
    
    if d == 2
        % --- Pauli 矩阵 ---
        basis_matrices(:,:,1) = [0 1; 1 0];       % Sigma X
        basis_matrices(:,:,2) = [0 -1i; 1i 0];    % Sigma Y
        basis_matrices(:,:,3) = [1 0; 0 -1];      % Sigma Z
    elseif d == 3
        % --- Gell-Mann 矩阵 ---
        basis_matrices(:,:,1) = [0 1 0; 1 0 0; 0 0 0];
        basis_matrices(:,:,2) = [0 -1i 0; 1i 0 0; 0 0 0];
        basis_matrices(:,:,3) = [1 0 0; 0 -1 0; 0 0 0];
        basis_matrices(:,:,4) = [0 0 1; 0 0 0; 1 0 0];
        basis_matrices(:,:,5) = [0 0 -1i; 0 0 0; 1i 0 0];
        basis_matrices(:,:,6) = [0 0 0; 0 0 1; 0 1 0];
        basis_matrices(:,:,7) = [0 0 0; 0 0 -1i; 0 1i 0];
        basis_matrices(:,:,8) = (1/sqrt(3)) * [1 0 0; 0 1 0; 0 0 -2];
    else
        error('当前程序仅支持 d=2 或 d=3');
    end

    % ================= 2. 从Bloch矢量提取测量基 =================
    % 总向量个数 n = 可观测量数 * 维度
    n = num_observables * d;
    measure = zeros(n, d);
    
    for j = 1:num_observables
        % 重构可观测量矩阵 H = sum(c_i * lambda_i)
        observable = zeros(d);
        vec = blochvector(:, j);
        for i = 1:expected_dim
            observable = observable + vec(i) * basis_matrices(:,:,i);
        end
        
        % 对角化获取特征向量 (eig 返回的 v 列向量已归一化)
        [v, ~] = eig(observable);
        
        % 将 d 个特征向量存入 measure 列表 (转置为行向量存储，方便后续处理)
        measure((j-1)*d + 1 : j*d, :) = v.'; 
    end

    % 边界情况处理
    if k == 0 || n == 0 || k > n
        max_value = 0;
        warning('组合参数无效: n=%d, k=%d', n, k);
        return;
    end

    % ================= 3. 预计算投影矩阵 =================
    % 计算 P_i = |v_i><v_i|
    projections = cell(n, 1);
    for i = 1:n
        v = measure(i, :).'; % 取出为列向量
        projections{i} = v * v'; % 外积
    end

    % ================= 4. 并行计算设置 =================
    % 计算总组合数
    total_combinations = nchoosek(n, k);
    
    % 获取并行资源
    try
        pool = gcp('nocreate');
        if isempty(pool)
            num_workers = feature('numCores'); % 默认使用物理核心数估计
        else
            num_workers = pool.NumWorkers;
        end
    catch
        num_workers = 1;
    end

    % 动态设置批次大小
    % 目标：每个 worker 处理约 50-100 个批次，既能负载均衡又能减少通信
    min_batches_per_worker = 50; 
    batch_size = ceil(total_combinations / (num_workers * min_batches_per_worker));
    
    % 限制批次大小范围，防止内存溢出或调度过于频繁
    batch_size = min(batch_size, 5000000); % 上限 500万
    batch_size = max(batch_size, 1000);    % 下限 1000
    
    num_batches = ceil(total_combinations / batch_size);

    % fprintf('开始计算 (Bloch模式): d=%d, n=%d, k=%d\n', d, n, k);
    % fprintf('总组合数: %d, 分批: %d (每批 %d)\n', total_combinations, num_batches, batch_size);
    
    total_tic = tic;
    batch_max_values = zeros(num_batches, 1);

    % ================= 5. 并行处理主循环 =================
    parfor batch_idx = 1:num_batches
        % 计算当前批次的全局索引范围
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, total_combinations);
        current_count = end_idx - start_idx + 1;
        
        local_max = 0;
        
        % --- 关键优化：直接计算起始组合 ---
        % 不再使用循环空跑，而是直接通过数学方法反解出第 start_idx 个组合
        current_comb = get_combination_from_rank(n, k, start_idx);
        
        for j = 1:current_count
            % 1. 累加投影矩阵
            sum_proj = zeros(d);
            for idx_ptr = 1:k
                vec_idx = current_comb(idx_ptr);
                sum_proj = sum_proj + projections{vec_idx};
            end
            
            % 2. 计算谱范数
            current_norm = norm(sum_proj);
            if current_norm > local_max
                local_max = current_norm;
            end
            
            % 3. 迭代到下一个组合 (仅在不是最后一次时计算)
            if j < current_count
                current_comb = get_next_combination(current_comb, n, k);
            end
        end
        
        batch_max_values(batch_idx) = local_max;
    end

    % ================= 6. 结果汇总 =================
    max_value = max(batch_max_values);
    total_time = toc(total_tic);
    
    % fprintf('计算完成。全局最大值: %.6f\n', max_value);
    % fprintf('总耗时: %.2f 秒\n', total_time);
end

% =========================================================================
%                             局部辅助函数
% =========================================================================

function comb = get_next_combination(comb, n, k)
% GET_NEXT_COMBINATION 生成字典序的下一个组合
% 查找最右侧可以增加的位置并进位
    i = k;
    while i > 0 && comb(i) == n - k + i
        i = i - 1;
    end
    
    if i > 0
        comb(i) = comb(i) + 1;
        for j = i+1:k
            comb(j) = comb(j-1) + 1;
        end
    end
end

function comb = get_combination_from_rank(n, k, rank)
% GET_COMBINATION_FROM_RANK 根据字典序排名直接计算组合 (Unranking)
% 避免了从头遍历产生的巨大开销
    comb = zeros(1, k);
    current_val = 1;
    
    for i = 1:k
        while true
            remaining_n = n - current_val;
            remaining_k = k - i;
            
            if remaining_n < remaining_k
                count = 0;
            else
                count = nchoosek(remaining_n, remaining_k);
            end
            
            if rank <= count
                comb(i) = current_val;
                current_val = current_val + 1;
                break;
            else
                rank = rank - count;
                current_val = current_val + 1;
            end
        end
    end
end
