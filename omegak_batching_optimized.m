function max_value = omegak_batching_optimized(k, base_array, d)
% OMEGAK_BATCHING_OPTIMIZED 计算 m 组测量集合中 k 个向量投影和的最大谱范数
%
% 输入:
%   k:          选取的向量数量 (标量)
%   base_array: (d x d x m) 三维数组，表示 m 组 d 维测量基
%   d:          希尔伯特空间维度
%
% 输出:
%   max_value:  所有可能的 k 个向量组合中，投影和矩阵的最大谱范数
%
% 逻辑说明:
%   1. 将所有基展平为 n 个向量。
%   2. 预计算每个向量的投影矩阵 P_i = |v_i><v_i|。
%   3. 将 C(n,k) 种组合分批次进行并行计算。
%   4. 使用组合数反解算法直接定位批次起点，避免由头遍历的开销。

    % ================= 1. 数据预处理 =================
    % 获取组数
    m = size(base_array, 3);
    
    % 将 (d x d x m) 转换为 (n x d) 的二维矩阵，每行一个向量
    % n = m * d 是总向量个数
    n = m * d;
    measure = zeros(n, d);
    for i = 1:m
        % 提取第 i 组基，放入对应行
        measure((i-1)*d+1 : i*d, :) = base_array(:, :, i);
    end

    % 边界检查
    if k == 0 || n == 0 || k > n
        max_value = 0;
        warning('输入参数 k 或 base_array 无效。');
        return;
    end

    % ================= 2. 预计算投影矩阵 =================
    % 预先计算 |v><v|，避免在循环内部重复进行外积运算
    % 使用 cell 数组存储，访问速度快
    projections = cell(n, 1);
    for i = 1:n
        v = measure(i, :).'; % 确保 v 是列向量
        projections{i} = v * v'; % 外积 (d x d)
    end

    % ================= 3. 并行计算参数设置 =================
    % 计算总组合数 nCk
    total_combinations = nchoosek(n, k);
    
    % 获取并行池信息以优化批次
    try
        pool = gcp('nocreate');
        if isempty(pool)
            % 默认不强制开启，由 parfor 自动处理，或者手动开启
            num_workers = feature('numCores'); 
        else
            num_workers = pool.NumWorkers;
        end
    catch
        num_workers = 1;
    end
    
    % 设置批次大小
    % 策略：为了负载均衡，批次不宜过大；为了减少通信开销，批次不宜过小。
    % 这里设置为每核处理约 50-100 个任务块，或设定一个固定的大数值
    target_tasks_per_worker = 100; 
    batch_size = ceil(total_combinations / (num_workers * target_tasks_per_worker));
    % 限制最大批次大小防止内存溢出或超时（例如上限 100万）
    batch_size = min(batch_size, 1000000); 
    batch_size = max(batch_size, 1000); % 至少 1000
    
    num_batches = ceil(total_combinations / batch_size);
    
    fprintf('开始计算: n=%d, k=%d, 总组合数=%d\n', n, k, total_combinations);
    fprintf('并行配置: 批次大小=%d, 总批次数=%d\n', batch_size, num_batches);

    % 记录批次结果的数组
    batch_results = zeros(num_batches, 1);
    
    total_tic = tic;

    % ================= 4. 并行处理主循环 =================
    parfor batch_idx = 1:num_batches
        % 计算当前批次的起始和结束索引（全局索引）
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, total_combinations);
        current_count = end_idx - start_idx + 1;
        
        local_max = 0;
        
        % --- 关键改进：直接计算当前批次的第一个组合 ---
        % 利用组合数反解算法，O(n) 复杂度，无需从第1个组合遍历到 start_idx
        current_comb = get_combination_from_rank(n, k, start_idx);
        
        % 批次内循环
        for j = 1:current_count
            % 1. 计算当前组合的投影和
            % 注意：在 parfor 中初始化变量最好显式进行
            sum_proj = zeros(d);
            for v_idx_ptr = 1:k
                % 获取组合中的向量索引
                vec_idx = current_comb(v_idx_ptr);
                sum_proj = sum_proj + projections{vec_idx};
            end
            
            % 2. 计算谱范数 (最大奇异值)
            current_norm = norm(sum_proj);
            
            if current_norm > local_max
                local_max = current_norm;
            end
            
            % 3. 生成下一个组合 (字典序)
            % 只有当不是本批次最后一个元素时才需要生成下一个
            if j < current_count
                current_comb = get_next_combination(current_comb, n, k);
            end
        end
        
        batch_results(batch_idx) = local_max;
    end

    % ================= 5. 结果汇总 =================
    max_value = max(batch_results);
    
    elapsed_time = toc(total_tic);
    fprintf('计算完成。全局最大值: %.6f (平均值 %.6f)\n', max_value, max_value/k);
    fprintf('总耗时: %.2f 秒\n', elapsed_time);
end

% =========================================================================
%                             局部辅助函数
% =========================================================================

function comb = get_next_combination(comb, n, k)
% GET_NEXT_COMBINATION 生成字典序的下一个组合
% 算法：从右向左寻找第一个可以增加的元素
    
    % 找到最右边可以增加的位置 i
    % 所谓“可以增加”，是指 comb(i) 的值还未达到其上限 (n - k + i)
    i = k;
    while i > 0 && comb(i) == n - k + i
        i = i - 1;
    end
    
    if i == 0
        % 已经是最后一个组合，无法继续（理论上外层循环控制不会进这里）
        return; 
    end
    
    % 当前位加 1
    comb(i) = comb(i) + 1;
    
    % 其右边的所有位依次递增
    for j = i+1:k
        comb(j) = comb(j-1) + 1;
    end
end

function comb = get_combination_from_rank(n, k, rank)
% GET_COMBINATION_FROM_RANK 根据字典序排名直接计算组合
% 输入: rank (1-based index)
% 输出: 对应的组合向量 [c1, c2, ..., ck]
% 
% 原理: 这是组合数系统的 "Unranking" 操作。
% 确定第一位 c1：我们检查如果 c1=1，剩下 k-1 个数从 n-1 里选，有多少种情况？
% 如果 rank <= C(n-1, k-1)，则 c1 确实是 1。
% 否则，rank 减去 C(n-1, k-1)，尝试 c1=2，以此类推。

    comb = zeros(1, k);
    current_val = 1;
    
    for i = 1:k
        % 我们需要选出第 i 个数，剩余还需要选 k-i 个数
        % 尝试 current_val 是否可以是当前位的数
        while true
            % 剩余可选的数有 n - current_val 个
            % 剩余需选位置数 k - i 个
            % 计算如果当前位选 current_val，后面有多少种组合
            
            remaining_n = n - current_val;
            remaining_k = k - i;
            
            % 防止 nchoosek 参数错误
            if remaining_n < remaining_k
                count = 0;
            else
                count = nchoosek(remaining_n, remaining_k);
            end
            
            if rank <= count
                % 说明当前位就是 current_val
                comb(i) = current_val;
                current_val = current_val + 1; % 下一位至少比当前大 1
                break;
            else
                % 说明当前位不是 current_val，或者是更大的数
                rank = rank - count;
                current_val = current_val + 1;
            end
        end
    end
end
