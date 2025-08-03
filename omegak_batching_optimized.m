%计算omegak，穷举所有可能组合，大维数多测量批次优化程序
function max_value = omegak_batching_optimized(k, measure, d) 
% 多核优化并行计算
% measure: (n x d) 矩阵，每行代表一个d维向量(测量基矢量)
% d: 希尔伯特空间维度
% k: 选取的向量数量

n = size(measure, 1);

% 边界情况处理
if k == 0 || n == 0 || k > n
    max_value = 0;
    return;
end

% 记录总开始时间
total_start_time = tic;

% ================= 预计算优化 =================
% 预计算所有向量的投影矩阵（避免在并行循环中重复计算）
% fprintf('预计算投影矩阵...\n');
projections = cell(n, 1);
for i = 1:n
    v = measure(i, :);
    projections{i} = v' * v;  % 外积 |v><v|
end

% ================ 组合数计算 =================
% 使用精确整数组合数计算
num_comb = nchoosek(n, k);
% fprintf('总组合数: %d\n', num_comb);

% =============== 并行参数设置 ================
% 获取可用CPU核心数
try
    pool = gcp('nocreate');
    if isempty(pool)
        % 尝试启动并行池（使用全部核心）
        try
            parpool;
            pool = gcp;
        catch
            pool = [];
        end
    end
    
    if isempty(pool)
        num_workers = 1;
    else
        num_workers = pool.NumWorkers;
    end
catch
    num_workers = 1;  % 回退到单核模式
end

% 动态设置批次大小, 可按照计算规模与核心数调整，保证充分利用核心以及内存
batch_size = min(400000000, ceil(num_comb / (1* num_workers)));
batch_size = max(batch_size, 1);  % 确保至少1个组合
num_batches = ceil(num_comb / batch_size);

% % 显示计算参数
% fprintf('优化并行计算参数:\n');
% fprintf('CPU核心数: %d\n', num_workers);
% fprintf('批次大小: %d\n', batch_size);
% fprintf('总批次数: %d\n', num_batches);

% ============== 精确计算批次范围 =============
% 确保批次范围不超出总组合数
batch_starts = zeros(1, num_batches);
batch_ends = zeros(1, num_batches);

for i = 1:num_batches
    batch_starts(i) = (i-1) * batch_size + 1;
    batch_ends(i) = min(i * batch_size, num_comb);
end

% ============== 并行处理主循环 ===============
% 为每个批次创建最大值存储数组
batch_max_values = zeros(1, num_batches);

% 外层并行循环 - 每个批次由独立工作进程处理
parfor batch_idx = 1:num_batches
    % ============ 临时变量初始化 ============
    % 获取当前批次范围
    start_idx = batch_starts(batch_idx);
    end_idx = batch_ends(batch_idx);
    current_batch_size = end_idx - start_idx + 1;
    
    % 初始化批次内最大值
    local_max = 0;
    
    % 创建组合生成器
    comb_gen = CombGenerator(n, k, start_idx, num_comb);
    
    % 处理当前批次的所有组合
    for j = 1:current_batch_size
        % ==== 获取下一个组合 ====
        comb = comb_gen.getNext();
        
        % ==== 计算投影和矩阵 ====
        sum_proj = zeros(d);  % 在parfor内部初始化
        
        % 计算当前组合的投影和
        for idx = 1:k
            v_idx = comb(idx);
            % 累加预计算的投影矩阵
            sum_proj = sum_proj + projections{v_idx};
        end
        
        % ==== 计算谱范数 ====
        current_norm = norm(sum_proj);  % 最大奇异值
        
        % 更新批次内最大值
        if current_norm > local_max
            local_max = current_norm;
        end
    end
    
    % 存储当前批次的最大值
    batch_max_values(batch_idx) = local_max;
    
    % 进度报告
    % completion = 100 * batch_idx / num_batches;
    % fprintf('批次 %d/%d 完成 (%.2f%%), 组合数: %d-%d, 批次最大值: %.6f\n', ...
    %         batch_idx, num_batches, completion, start_idx, end_idx, local_max);
end


% 在所有批次完成后，找到全局最大值
max_value = max(batch_max_values);

% 计算总时间
total_time = toc(total_start_time);

% fprintf('\n计算完成!\n');
% fprintf('全局最大值: %.6f\n', max_value/k);
% fprintf('总计算时间: %.2f 分钟\n', total_time/60);