function H = calc_shannon_entropy(p)
% CALC_SHANNON_ENTROPY 计算香农熵 (以比特为单位, log2)
% 输入: p - 概率向量 (行向量或列向量)
    
    % 1. 确保是列向量
    p = p(:);
    
    % 2. (可选) 归一化检查：如果输入不是概率分布(和不为1)，则归一化
    % 如果你确定输入已经是归一化的，可以注释掉下面这行
    % if abs(sum(p) - 1) > 1e-10
    %     p = p / sum(p);
    %     % warning('输入向量已自动归一化');
    % end

    % 3. 剔除 0 元素 (防止 0*log(0) = NaN)
    % 只有大于 0 的元素才参与计算
    p_nz = p(p > 0);
    
    % 4. 计算熵
    if isempty(p_nz)
        H = 0;
    else
        H = -sum(p_nz .* log2(p_nz));
    end
end
