function [bestSolution, bestEnergy, history] = crossEntropyOptimizerWerner(d, n, sampleSize, eliteSize, maxIterations, smoothingFactor, minStd)
    % 交叉熵优化算法实现
    % 输入参数:
    %   d: 维度
    %   n: 向量数量
    %   sampleSize: 每代样本数量
    %   eliteSize: 精英样本数量
    %   maxIterations: 最大迭代次数
    %   smoothingFactor: 平滑更新因子 (0.1-0.9)
    %   minStd: 最小标准差阈值
    
    % === 添加时间记录 ===
    totalStartTime = tic;
    iterTimes = zeros(maxIterations, 1);  % 存储每次迭代耗时
    
    % 1. 初始化分布参数
    mu = randn(d^2-1, n);  % 均值矩阵
    sigma = 1.0 * ones(d^2-1, n);  % 标准差矩阵
    
    % 归一化初始均值
    for i = 1:n
        mu(:, i) = mu(:, i) / norm(mu(:, i));
    end
    
    % 记录最优解
    bestEnergy = inf;
    bestSolution = [];
    history = struct('bestEnergies', [], 'meanEnergies', [], 'avgStds', [], 'eliteEnergies', []);
    
    fprintf('开始交叉熵优化 (n=%d, 样本数=%d, 精英数=%d)\n', n, sampleSize, eliteSize);
    
    % 主优化循环
    for iter = 1:maxIterations
        iterStart = tic;  % 记录迭代开始时间
        
        % 2. 从当前分布生成样本
        solutions = cell(sampleSize, 1);
        energies = zeros(sampleSize, 1);
        
        parfor s = 1:sampleSize  % 使用并行计算加速
            % 生成候选解
            candidate = mu + sigma .* randn(d^2-1, n);

            % 初始化测量基
            % candidate = zeros(d*n,d);
            % 生成Haar均匀分布的测量基
            % for i = 1:n
            %     candidate((i-1)*d+1:d*i,:) = orth(mu((i-1)*d+1:d*i,:) + sigma((i-1)*d+1:d*i,:).*randn(d,d,"like",1i)).';
            % end
            
            % 归一化所有向量
            for i = 1:n
                candidate(:, i) = candidate(:, i) / norm(candidate(:, i));
            end

            % 计算能量
            % energy = computeTheta(candidate, d)/n
            % energy = compute_max_norm_optimized(candidate, d)/n; % omegak_batching_optimized(n, candidate, d) / n; % Isotropic 态
            energy = omegak_batching_optimized_Bloch((d-1)*n, candidate, d) / ((d-1)*n); % Werner 态
            
            solutions{s} = candidate;
            energies(s) = energy;
        end
        
        % 3. 选择精英样本
        [sortedEnergies, sortIdx] = sort(energies);
        eliteSolutions = solutions(sortIdx(1:eliteSize));
        eliteEnergies = sortedEnergies(1:eliteSize);
        
        % 4. 更新全局最优解
        if eliteEnergies(1) < bestEnergy
            bestEnergy = eliteEnergies(1);
            bestSolution = eliteSolutions{1};
            fprintf('迭代 %d: 发现新最优解! 能量=%.8f\n', iter, bestEnergy);
        end
        
        % 5. 更新分布参数
        % 5.1 计算精英样本的均值和标准差
        eliteTensor = cat(3, eliteSolutions{:});
        newMu = mean(eliteTensor, 3);
        newSigma = std(eliteTensor, 0, 3);  % 0表示使用n-1计算标准差
        
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
        iterTimes(iter) = toc(iterStart);  % 本次迭代耗时
        elapsedTime = toc(totalStartTime); % 总耗时
        avgIterTime = sum(iterTimes(1:iter)) / iter;
        remainingTime = (maxIterations - iter) * avgIterTime;
        
        % 显示进度（添加时间信息）
        avgStd = mean(sigma(:));
        fprintf('迭代 %d/%d: 最优能量=%.8f, 精英平均=%.8f, 平均标准差=%.4f\n',...
                iter, maxIterations, bestEnergy, mean(eliteEnergies), avgStd);
        fprintf('    时间: 已用 %s, 估计剩余 %s\n', ...
                formatTime(elapsedTime), formatTime(remainingTime));
        
        % 绘制进度
        if mod(iter, 10) == 0 || iter == maxIterations
            plotCrossEntropyProgress(d, history, iter);
        end
        
        % 提前终止条件：标准差足够小
        if avgStd < 1.1 * minStd
            fprintf('提前终止: 平均标准差已接近最小值 (%.4f < %.4f)\n', avgStd, 1.1*minStd);
            break;
        end
    end
    
    % 最终结果显示（添加总耗时）
    totalTime = toc(totalStartTime);
    fprintf('\n优化完成! 迭代次数: %d/%d, 总耗时: %s\n', ...
            iter, maxIterations, formatTime(totalTime));
    fprintf('平均Ω_N = %.8f\n', bestEnergy);
    fprintf('w = %.8f\n', (d*bestEnergy-1)*(d-1)); % Werner 态
    
    % 绘制最终结果
    plotFinalResults(d, history, bestEnergy);
end

% === 新增函数：格式化时间显示 ===
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

% 随机生成Haar均匀幺正矩阵
% function U = random_haar_unitary(n)
%     % 生成 n x n 随机复高斯矩阵
%     Z = (randn(n) + 1i*randn(n))/sqrt(2);
% 
%     % 进行 QR 分解
%     [Q, R] = qr(Z);
% 
%     % 提取对角元素的相位
%     phases = diag(R)./abs(diag(R));
% 
%     % 构造相位校正矩阵
%     D = diag(phases);
% 
%     % 计算随机幺正矩阵
%     U = Q * D;
% end
% 
% function U = random_hs_unitary(dim, sigma, U0)
%     % 生成 dim x dim 随机幺正矩阵，服从以 U0 为中心、宽度为 sigma 的 Hilbert-Schmidt 分布
%     % 默认值设置
%     if nargin < 3
%         U0 = eye(dim); % 默认为单位矩阵
%     end
%     if nargin < 2
%         sigma = 1; % 默认分布宽度
%     end
% 
%     % 生成复高斯随机矩阵
%     G = (randn(dim) + 1i*randn(dim)) * (sigma/sqrt(2));
% 
%     % 计算 Hermitian 矩阵
%     H = U0' * G; % 初始偏移
%     H = (H + H')/2; % 确保 Hermitian
% 
%     % 通过指数映射生成幺正矩阵
%     U = U0 * expm(1i * H);
% end


function plotCrossEntropyProgress(d, history, currentIter)
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
    title('能量变化曲线');
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
    wValues = (d*history.bestEnergies - 1)*(d-1);
    plot(iterRange, wValues, 'c-', 'LineWidth', 2);
    title('w值变化');
    xlabel('迭代次数');
    ylabel('w值');
    grid on;
    
    subplot(2,2,4);
    improvement = [0, diff(history.bestEnergies)];
    improvement(improvement > 0) = 0;  % 只保留负值（改进）
    bar(iterRange, improvement, 'b');
    title('每次迭代的改进量');
    xlabel('迭代次数');
    ylabel('能量改进');
    grid on;
    
    drawnow;
end

function plotFinalResults(d, history, bestEnergy)
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
    wValues = (d*history.bestEnergies - 1)*(d-1);
    plot(iterRange, wValues, 'm-', 'LineWidth', 2);
    finalW = (d*bestEnergy-1)*(d-1);
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
