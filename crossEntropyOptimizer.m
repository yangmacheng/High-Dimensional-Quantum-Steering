function [bestSolution, bestEnergy, history] = crossEntropyOptimizer(targetType, d, n, sampleSize, eliteSize, maxIterations, smoothingFactor, minStd)
    % CROSSENTROPYOPTIMIZER 交叉熵优化器 (合并 Isotropic 和 Werner 模式)
    % 
    % 输入参数:
    %   targetType:      优化目标类型, 'iso' (Isotropic态) 或 'werner' (Werner态)
    %   d:               维度
    %   n:               向量数量 (Bloch矢量列数)
    %   sampleSize:      每代样本数量
    %   eliteSize:       精英样本数量
    %   maxIterations:   最大迭代次数
    %   smoothingFactor: 平滑更新因子 (0.1-0.9)
    %   minStd:          最小标准差阈值
    
    % === 0. 根据类型配置参数 ===
    targetType = lower(targetType);
    
    % 默认配置
    if strcmp(targetType, 'iso')
        % Isotropic 态配置
        calc_k = n;                    % omegak 的 k 参数
        norm_factor = n;               % 能量归一化分母
        init_sigma_val = 3.0;          % 初始标准差
        w_func = @(E) (d*E - 1)/(d - 1); % Isotropic 的 w 计算公式
        fprintf('模式: Isotropic State Optimization (最小化能量)\n');
        
    elseif strcmp(targetType, 'werner')
        % Werner 态配置
        calc_k = (d - 1) * n;          % omegak 的 k 参数
        norm_factor = (d - 1) * n;     % 能量归一化分母
        init_sigma_val = 1.0;          % 初始标准差
        w_func = @(E) (d*E - 1)*(d - 1); % Werner 的 w 计算公式 (保留原代码逻辑)
        fprintf('模式: Werner State Optimization (最小化能量)\n');
        
    else
        error('未知的 targetType。请使用 ''iso'' 或 ''werner''。');
    end

    % === 添加时间记录 ===
    totalStartTime = tic;
    iterTimes = zeros(maxIterations, 1);
    
    % 1. 初始化分布参数
    % blochvector 维度为 d^2-1
    dim_bloch = d^2 - 1;
    mu = randn(dim_bloch, n);        % 均值矩阵
    sigma = init_sigma_val * ones(dim_bloch, n); % 标准差矩阵
    
    % 归一化初始均值
    for i = 1:n
        mu(:, i) = mu(:, i) / norm(mu(:, i));
    end
    
    % 记录最优解 (初始化为无穷大，因为是最小化问题)
    bestEnergy = inf;
    bestSolution = [];
    history = struct('bestEnergies', [], 'meanEnergies', [], 'avgStds', [], 'eliteEnergies', []);
    
    fprintf('开始交叉熵优化 (n=%d, d=%d, 样本数=%d, 精英数=%d)\n', n, d, sampleSize, eliteSize);
    
    % 主优化循环
    for iter = 1:maxIterations
        iterStart = tic;
        
        % 2. 从当前分布生成样本
        solutions = cell(sampleSize, 1);
        energies = zeros(sampleSize, 1);
        
        % 将变量提取到 parfor 之外以减少通信开销
        current_mu = mu;
        current_sigma = sigma;
        
        parfor s = 1:sampleSize
            % 生成候选解
            candidate = current_mu + current_sigma .* randn(dim_bloch, n);

            % 归一化所有向量 (Bloch 矢量需在球面上)
            for i = 1:n
                candidate(:, i) = candidate(:, i) / norm(candidate(:, i));
            end

            % 计算能量 (调用前面的 omegak_batching_optimized_Bloch)
            % 注意：根据不同模式传入不同的 calc_k
            raw_val = omegak_batching_optimized_Bloch(calc_k, candidate, d);
            
            % 归一化
            energy = raw_val / norm_factor;
            
            solutions{s} = candidate;
            energies(s) = energy;
        end
        
        % 3. 选择精英样本 (最小化：从小到大排序)
        [sortedEnergies, sortIdx] = sort(energies, 'ascend');
        eliteSolutions = solutions(sortIdx(1:eliteSize));
        eliteEnergies = sortedEnergies(1:eliteSize);
        
        % 4. 更新全局最优解
        if eliteEnergies(1) < bestEnergy
            bestEnergy = eliteEnergies(1);
            bestSolution = eliteSolutions{1};
            % 计算当前的 w 用于显示
            currentW = w_func(bestEnergy);
            fprintf('迭代 %d: 发现新最优解! 能量=%.8f, w=%.8f\n', iter, bestEnergy, currentW);
        end
        
        % 5. 更新分布参数
        % 5.1 计算精英样本的均值和标准差
        eliteTensor = cat(3, eliteSolutions{:});
        newMu = mean(eliteTensor, 3);
        newSigma = std(eliteTensor, 0, 3);  % 0表示使用 N-1 计算标准差
        
        % 5.2 平滑更新
        mu = smoothingFactor * newMu + (1 - smoothingFactor) * mu;
        sigma = smoothingFactor * newSigma + (1 - smoothingFactor) * sigma;
        
        % 5.3 应用标准差下限
        sigma = max(sigma, minStd);
        
        % 记录历史数据
        history.bestEnergies(iter) = bestEnergy;
        history.meanEnergies(iter) = mean(energies);
        history.avgStds(iter) = mean(sigma(:));
        history.eliteEnergies(iter) = mean(eliteEnergies);
        
        % === 计算时间信息 ===
        iterTimes(iter) = toc(iterStart);
        elapsedTime = toc(totalStartTime);
        avgIterTime = sum(iterTimes(1:iter)) / iter;
        remainingTime = (maxIterations - iter) * avgIterTime;
        
        % 显示进度
        avgStd = mean(sigma(:));
        fprintf('迭代 %d/%d: 最优=%.8f, 精英均值=%.8f, Std=%.4f\n',...
                iter, maxIterations, bestEnergy, mean(eliteEnergies), avgStd);
        fprintf('    时间: 已用 %s, 估计剩余 %s\n', ...
                formatTime(elapsedTime), formatTime(remainingTime));
        
        % 绘制进度 (传入 w_func 以正确计算 w)
        if mod(iter, 10) == 0 || iter == maxIterations
            plotCrossEntropyProgress(d, history, iter, w_func);
        end
        
        % 提前终止条件
        if avgStd < 1.1 * minStd
            fprintf('提前终止: 平均标准差已接近最小值 (%.4f < %.4f)\n', avgStd, 1.1*minStd);
            break;
        end
    end
    
    % 最终结果显示
    totalTime = toc(totalStartTime);
    finalW = w_func(bestEnergy);
    
    fprintf('\n优化完成! 模式: %s\n', targetType);
    fprintf('迭代次数: %d/%d, 总耗时: %s\n', iter, maxIterations, formatTime(totalTime));
    fprintf('最终平均 Ω_N = %.8f\n', bestEnergy);
    fprintf('最终 w = %.8f\n', finalW);
    
    % 绘制最终结果
    plotFinalResults(d, history, bestEnergy, w_func);
end

% === 辅助函数：格式化时间显示 ===
function str = formatTime(seconds)
    if seconds < 60
        str = sprintf('%.1f秒', seconds);
    elseif seconds < 3600
        minutes = floor(seconds/60);
        seconds = mod(seconds, 60);
        str = sprintf('%d分%.1f秒', minutes, seconds);
    elseif seconds < 86400
        hours = floor(seconds/3600);
        minutes = floor(mod(seconds, 3600)/60);
        seconds = mod(seconds, 60);
        str = sprintf('%d小时%d分%.1f秒', hours, minutes, seconds);
    else
        days = floor(seconds/86400);
        hours = floor(mod(seconds, 86400)/3600);
        minutes = floor(mod(seconds, 3600)/60);
        seconds = mod(seconds, 60);
        str = sprintf('%d天%d小时%d分%.1f秒', days, hours, minutes, seconds);
    end
end

% === 绘图函数：接收 w_func 句柄 ===
function plotCrossEntropyProgress(d, history, currentIter, w_func)
    % 绘制交叉熵优化进度
    figure(1);
    set(gcf, 'Name', '交叉熵优化进度', 'Position', [100, 100, 1200, 800]);
    
    iterRange = 1:currentIter;
    
    subplot(2,2,1);
    plot(iterRange, history.bestEnergies, 'b-', 'LineWidth', 2);
    hold on;
    plot(iterRange, history.eliteEnergies, 'g--', 'LineWidth', 1.5);
    plot(iterRange, history.meanEnergies, 'r:', 'LineWidth', 1.5);
    hold off;
    title('能量变化曲线 (Minimization)');
    xlabel('迭代次数');
    ylabel('能量');
    legend('最优能量', '精英平均能量', '总样本平均能量', 'Location', 'best');
    grid on;
    
    subplot(2,2,2);
    semilogy(iterRange, history.avgStds, 'm-', 'LineWidth', 2);
    title('标准差衰减 (对数尺度)');
    xlabel('迭代次数');
    ylabel('平均标准差');
    grid on;
    
    subplot(2,2,3);
    % 使用传入的函数句柄计算 w
    wValues = arrayfun(w_func, history.bestEnergies);
    plot(iterRange, wValues, 'c-', 'LineWidth', 2);
    title('w值变化');
    xlabel('迭代次数');
    ylabel('w值');
    grid on;
    
    subplot(2,2,4);
    % 改进量 (能量下降量)
    improvement = [0, -diff(history.bestEnergies)]; % 取负，因为能量下降是改进
    improvement(improvement < 0) = 0;
    bar(iterRange, improvement, 'b');
    title('每次迭代的改进量 (能量下降)');
    xlabel('迭代次数');
    ylabel('能量改进');
    grid on;
    
    drawnow;
end

function plotFinalResults(d, history, bestEnergy, w_func)
    % 绘制最终结果
    figure(2);
    set(gcf, 'Name', '最终优化结果', 'Position', [200, 200, 1000, 400]);
    
    iterRange = 1:length(history.bestEnergies);
    
    subplot(1,3,1);
    plot(iterRange, history.bestEnergies, 'b-', 'LineWidth', 2);
    yline(bestEnergy, 'r--', 'LineWidth', 1.5);
    title(sprintf('最优能量变化 (最终: %.8f)', bestEnergy));
    xlabel('迭代次数');
    ylabel('能量');
    grid on;
    
    subplot(1,3,2);
    % 使用传入的函数句柄计算 w
    wValues = arrayfun(w_func, history.bestEnergies);
    finalW = w_func(bestEnergy);
    
    plot(iterRange, wValues, 'm-', 'LineWidth', 2);
    yline(finalW, 'r--', 'LineWidth', 1.5);
    title(sprintf('w值变化 (最终: %.8f)', finalW));
    xlabel('迭代次数');
    ylabel('w值');
    grid on;
    
    subplot(1,3,3);
    semilogy(iterRange, history.avgStds, 'g-', 'LineWidth', 2);
    title('标准差衰减');
    xlabel('迭代次数');
    ylabel('平均标准差 (对数)');
    grid on;
end
