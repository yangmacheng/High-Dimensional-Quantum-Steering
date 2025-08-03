%计算omegak，穷举所有可能组合，大维数多测量批次优化程序，使用Bloch表示参数化
function max_value = omegak_batching_optimized_Bloch(k, blochvector, d) 
% 多核优化并行计算
% d: 希尔伯特空间维度
% k: 选取的向量数量

%Bloch参数化表示
%blochvector是一个(d^2-1) x n的矩阵数组，每一列表示可观测的d^2-1维Bloch矢量

% 获取Bloch矢量总数，或测量总数
[~,num] = size(blochvector);

% 定义 Pauli 矩阵
sigam = zeros(2,2,3); % 预分配2×2×3数组
% 单位矩阵
% I = [1 0; 0 1];

% Pauli X 矩阵 (比特翻转)
sigma(:,:,1) = [0 1; 1 0];

% Pauli Y 矩阵
sigma(:,:,2) = [0 -1i; 1i 0];

% Pauli Z 矩阵 (相位翻转)
sigma(:,:,3) = [1 0; 0 -1];

% 生成SU(3)群的8个盖尔曼矩阵（3×3）
lambda = zeros(3,3,8); % 预分配3×3×8数组

% λ1
lambda(:,:,1) = [0  1  0; 
                 1  0  0; 
                 0  0  0];

% λ2
lambda(:,:,2) = [0 -1i 0; 
                 1i 0  0; 
                 0  0  0];

% λ3
lambda(:,:,3) = [1  0  0; 
                 0 -1  0; 
                 0  0  0];

% λ4
lambda(:,:,4) = [0  0  1; 
                 0  0  0; 
                 1  0  0];

% λ5
lambda(:,:,5) = [0  0 -1i; 
                 0  0  0; 
                 1i 0  0];

% λ6
lambda(:,:,6) = [0  0  0; 
                 0  0  1; 
                 0  1  0];

% λ7
lambda(:,:,7) = [0  0  0; 
                 0  0 -1i; 
                 0 1i 0];

% λ8 (需归一化)
lambda(:,:,8) = (1/sqrt(3)) * [1  0  0; 
                                0  1  0; 
                                0  0 -2];

if d == 2
    measure = zeros(2*num,2);
    for j = 1:num
        observable = 0;
        for i = 1:3
            observable = observable+blochvector(i,j)*sigma(:,:,i);
        end
        [v,~] = eig(observable);
        measure(1+2*(j-1):2*j,:) = v.';
    end
else
    measure = zeros(3*num,3);
    for j = 1:num
        observable = 0;
        for i = 1:8
            observable = observable+blochvector(i,j)*lambda(:,:,i);
        end
        [v,~] = eig(observable);
        measure(1+3*(j-1):3*j,:) = v.';
    end
end


% 如果直接给出测量基，可以注释掉前面的Bloch参数化代码，直接运行以下代码
% 获取测量基矢量总数
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