function [optimal_matrices, optimal_prob_dist, optimal_value, history] = minimization_entropy( ...
    d, rho, base_array, max_iter, n_samples, elite_frac, smoothing, minStd)
% MINIMIZATION_ENTROPY 基于交叉熵方法(CEM)寻找最小化输出熵的广义量子算符
%
% 物理背景:
%   该函数用于寻找一个最优的量子变换操作 U（可以是幺正变换或反幺正变换），
%   作用于测量基 base_array 上。
%   目标是使得初始态 rho 在变换后的测量基下，其测量概率分布的 Shannon 熵最小。
%
% 算法原理 (Cross-Entropy Method):
%   1. 参数化: 将算符参数化为 d^2+1 维向量。前 d^2 维对应幺正变换参数，
%      最后 1 维用于控制是否施加反幺正操作 (Complex Conjugate)。
%   2. 采样: 根据高斯分布 N(mu, sigma) 生成参数样本。
%   3. 评估: 生成算符 U，变换测量基，计算 rho 在新基下的测量熵。
%   4. 精英选择: 选出熵值最低的前 elite_frac 比例的样本。
%   5. 更新: 利用精英样本更新高斯分布的 mu 和 sigma (引入平滑因子)。
%   6. 迭代: 重复直到收敛。
%
% 输入参数:
%   d           - 标量, 希尔伯特空间的维度。
%   rho         - d^2 x d^2 (或对应维度) 矩阵, 量子态密度矩阵。
%   base_array  - 3D 数组 (d x d x M), 初始测量基集合。
%   max_iter    - 标量, 最大迭代次数。
%   n_samples   - 标量, 每一代生成的候选样本数量。
%   elite_frac  - 标量, 精英样本比例 (0 < elite_frac <= 1)。
%   smoothing   - 标量, 平滑更新因子 (0 < smoothing <= 1)。
%   minStd      - (可选) 标量, 最小标准差阈值, 默认 1e-4。
%
% 输出参数:
%   optimal_matrices - 结构体, 包含最优广义算符信息:
%                      .U       : d x d 幺正矩阵
%                      .is_anti : 逻辑值 (true/false)。
%                                 如果为 true，表示物理操作为 base_array -> pagemtimes(conj(base_array), 'none', U, 'transpose')
%                                 如果为 false，表示物理操作为 base_array -> pagemtimes(base_array, 'none', U, 'transpose')
%   optimal_value    - 标量, 达到的最小 Shannon 熵值。
%   optimal_prob_dist - 向量, 对应于最优熵值的测量概率分布。
%   history          - 结构体, 收敛历史。
%
% 外部依赖:
%   - generate_general_operator(d, params): 生成幺正或反幺正矩阵。
%   - calc_quantum_prob_aggregation(rho, base_alice, base_bob): 计算概率。
%   - calc_shannon_entropy(probs): 计算熵。
% -------------------------------------------------------------------------

% === 1. 参数校验与初始化 ===

if nargin < 8
    minStd = 1e-4;
end

totalStartTime = tic;
iterTimes = zeros(max_iter, 1);

% 计算参数空间维度
% d^2 个参数用于生成幺正矩阵，1 个参数用于判断是否进行共轭（反幺正）
total_param_dim = d^2 + 1;

% 初始化高斯分布参数
mu = zeros(1, total_param_dim);         
sigma = 4.0 * ones(1, total_param_dim); 

% 计算每一代的精英样本绝对数量
eliteSize = max(1, round(n_samples * elite_frac));

% 初始化最优解
best_value = inf;              
% 初始化最优解结构体
optimal_matrices = struct('U', eye(d), 'is_anti', false);
% 初始化最优概率分布
optimal_prob_dist = [];

% 初始化历史记录
history = struct('bestEnergies', zeros(max_iter, 1), ...
    'meanEnergies', zeros(max_iter, 1), ...
    'avgStds', zeros(max_iter, 1), ...
    'eliteEnergies', zeros(max_iter, 1));

fprintf('=== 开始熵最小化优化 (CEM算法) ===\n');
fprintf('维度 d=%d | 样本数 n=%d | 精英数=%d | 参数维数=%d\n', ...
    d, n_samples, eliteSize, total_param_dim);


% === 2. 主优化循环 ===
for iter = 1:max_iter
    iterStart = tic;

    % --- 2.1 样本生成 ---
    % 从当前分布 N(mu, sigma) 中生成参数样本
    params_samples = mu + sigma .* randn(n_samples, total_param_dim);

    % 临时存储结构体，存储最优广义变换信息及概率分布
    ops_storage = cell(n_samples, 1); 
    values = zeros(n_samples, 1); 

    % --- 并行评估 ---
    parfor s = 1:n_samples
        current_params = params_samples(s, :);

        % 生成算符 (幺正 或 反幺正)
        [U, is_anti] = generate_general_operator(d, current_params);
        
        % 应用变换
        if is_anti
            % 反幺正: Conj(Data) * U.'
            base_alice = pagemtimes(conj(base_array), 'none', U, 'transpose');
        else
            % 幺正: Data * U.'
            base_alice = pagemtimes(base_array, 'none', U, 'transpose');
        end

        % 计算测量概率分布
        prob_dist = calc_quantum_prob_aggregation(rho, base_alice, base_array);

        % 计算目标函数 (Shannon 熵)
        val = calc_shannon_entropy(prob_dist);

        % 将 U, is_anti 和 prob_dist 打包存入结构体
        op_struct = struct();
        op_struct.U = U;
        op_struct.is_anti = is_anti;
        op_struct.prob_dist = prob_dist; % 存储对应的概率分布
        
        ops_storage{s} = op_struct;
        values(s) = val;
    end

    % --- 2.2 精英选择 ---
    [sortedValues, sortIdx] = sort(values);

    eliteIndices = sortIdx(1:eliteSize);
    eliteParams = params_samples(eliteIndices, :);
    eliteValues = sortedValues(1:eliteSize);

    % --- 2.3 更新全局最优解 ---
    if eliteValues(1) < best_value
        best_value = eliteValues(1);

        % 提取最优的结构体
        best_struct = ops_storage{eliteIndices(1)};

        % 更新输出变量
        optimal_matrices.U = best_struct.U;
        optimal_matrices.is_anti = best_struct.is_anti;
        optimal_prob_dist = best_struct.prob_dist; % 提取概率分布
       
        % 打印调试信息，显示当前找到的是不是反幺正
        typeStr = 'Unitary';
        if optimal_matrices.is_anti
            typeStr = 'Anti-Unitary';
        end
        fprintf('>> 迭代 %d: 新最优 (%s)! 熵=%.8f\n', iter, typeStr, best_value);
    end

    % --- 2.4 更新分布参数 ---
    newMu = mean(eliteParams, 1);
    newSigma = std(eliteParams, 0, 1);

    % 平滑更新
    mu = smoothing * newMu + (1 - smoothing) * mu;
    sigma = smoothing * newSigma + (1 - smoothing) * sigma;
    
    % 强制最小标准差
    sigma = max(sigma, minStd);

    % --- 2.5 记录与显示 ---
    history.bestEnergies(iter) = best_value;
    history.meanEnergies(iter) = mean(values);
    history.avgStds(iter) = mean(sigma(:));
    history.eliteEnergies(iter) = mean(eliteValues);

    iterTimes(iter) = toc(iterStart);
    avgStd = mean(sigma(:));

    if mod(iter, 10) == 0 || iter == 1
         fprintf('迭代 %03d/%d: 最优=%.6f, 精英均值=%.6f, Std=%.4f, 耗时=%.2fs\n',...
            iter, max_iter, best_value, mean(eliteValues), avgStd, iterTimes(iter));
    end

    % --- 2.6 绘图 ---
    if mod(iter, 10) == 0 || iter == max_iter
        plotEntropyProgress(history, iter);
    end

    % --- 2.7 提前终止 ---
    if avgStd < 1.1 * minStd
        fprintf('\n*** 收敛终止: 平均标准差 (%.4f) 接近阈值 ***\n', avgStd);
        % 截断多余的历史记录
        history.bestEnergies = history.bestEnergies(1:iter);
        history.meanEnergies = history.meanEnergies(1:iter);
        history.avgStds = history.avgStds(1:iter);
        history.eliteEnergies = history.eliteEnergies(1:iter);
        break;
    end
end

% === 3. 结束处理 ===
totalTime = toc(totalStartTime);
optimal_value = best_value;

fprintf('\n=== 优化完成 ===\n');
fprintf('总耗时: %s\n', formatTime(totalTime));
fprintf('最终熵值: %.8f\n', best_value);

plotFinalEntropyResults(history, best_value);

end

% =========================================================================
%                               辅助函数
% =========================================================================

function str = formatTime(seconds)
    if seconds < 60
        str = sprintf('%.1f秒', seconds);
    elseif seconds < 3600
        str = sprintf('%.1f分', seconds/60);
    else
        str = sprintf('%.1f小时', seconds/3600);
    end
end

function plotEntropyProgress(history, currentIter)
    % 如果没有图形窗口，创建一个
    if isempty(findobj('Type','Figure','Name','熵最小化优化进度'))
        figure('Name', '熵最小化优化进度', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 400]);
    else
        set(0, 'CurrentFigure', findobj('Type','Figure','Name','熵最小化优化进度'));
    end

    iterRange = 1:currentIter;

    subplot(1, 2, 1);
    plot(iterRange, history.bestEnergies(1:currentIter), 'b-', 'LineWidth', 1.5); hold on;
    plot(iterRange, history.eliteEnergies(1:currentIter), 'g--', 'LineWidth', 1);
    plot(iterRange, history.meanEnergies(1:currentIter), 'r:', 'LineWidth', 1);
    hold off;
    title('熵收敛曲线'); xlabel('Iter'); ylabel('Entropy');
    legend('Best', 'Elite Mean', 'Pop Mean'); grid on; xlim([1, max(2, currentIter)]);

    subplot(1, 2, 2);
    semilogy(iterRange, history.avgStds(1:currentIter), 'k-', 'LineWidth', 1.5);
    title('参数分布收敛 (Std)'); xlabel('Iter'); ylabel('Avg Std (Log)');
    grid on; xlim([1, max(2, currentIter)]);
    
    drawnow limitrate;
end

function plotFinalEntropyResults(history, bestValue)
% plotFinalEntropyResults 优化结束后绘制最终总结图
figure(102);
set(gcf, 'Name', '最终优化结果摘要', 'NumberTitle', 'off', 'Position', [200, 200, 800, 400]);

iterRange = 1:length(history.bestEnergies);

% 左图: 最终收敛轨迹
subplot(1, 2, 1);
plot(iterRange, history.bestEnergies, 'b-', 'LineWidth', 2);
yline(bestValue, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Min: %.6f', bestValue));
title('最终收敛轨迹');
xlabel('迭代次数');
ylabel('Shannon 熵');
grid on;

% 右图: 最终参数收敛情况
subplot(1, 2, 2);
semilogy(iterRange, history.avgStds, 'g-', 'LineWidth', 2);
title('参数不确定性收敛');
xlabel('迭代次数');
ylabel('平均标准差 (Log)');
grid on;
end
