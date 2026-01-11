function sorted_Upsilon_total = calc_quantum_prob_aggregation(rho, base_alice, base_bob)
% CALC_QUANTUM_PROB_AGGREGATION 计算量子测量概率并聚合排序
% 
% 输入:
%   rho: (d^2 x d^2) 密度矩阵
%   base_alice, base_bob: Alice 和 Bob 的 (d x d x m) 测量基数组
%   U: 幺正或反幺正矩阵。
%
% 更新说明:
%   使用了循环对角线求和的向量化实现，对应公式:
%   j = (i + k - 2) mod d + 1

    [d_rows, d_cols, m] = size(base_bob);
    d = d_rows;
    
    % 维度检查
    if d_rows ~= d_cols; error('基矢量必须是 d x d'); end
    if size(rho, 1) ~= d^2; error('rho 维度错误'); end

    Upsilon_total = [];

    for mu = 1:m
        % 提取第 mu 组基 (转置使得列向量为基矢量 |v_i>)
        Basis_alice_mu = squeeze(base_alice(:, :, mu)).';
        Basis_bob_mu = squeeze(base_bob(:, :, mu)).';
        
        % 1. 计算联合概率矩阵 P (d x d)
        P = zeros(d, d);
        for i = 1:d
            for j = 1:d
                vec_kron = kron(Basis_alice_mu(:, i), Basis_bob_mu(:, j));
                P(i, j) = real(vec_kron' * rho * vec_kron);
            end
        end
        
        % 2. 聚合操作 (Aggregation) - 向量化实现
        Upsilon = zeros(d, 1);
        
        % 数学原理：
        % I_k 对应的是矩阵 P 中偏移量为 (k-1) 的循环对角线。
        % 我们将 P 的列向左循环移动 (k-1) 位，原本在循环对角线上的元素
        % 就会移动到主对角线上。
        
        for k = 1:d
            shift_amount = -(k - 1); % 负号表示向左移，对应列索引增加
            
            % circshift(A, [r, c]) 表示行移r，列移c
            P_shifted = circshift(P, [0, shift_amount]);
            
            % 求对角线之和
            Upsilon(k) = sum(diag(P_shifted));
        end
        
        % 3. 收集结果
        Upsilon_total = [Upsilon_total; Upsilon];
    end
    
    % 降序排列
    sorted_Upsilon_total = sort(Upsilon_total, 'descend');
    % 取前 m*(d-1) 项
    % sorted_Upsilon_part = sorted_Upsilon_total(1:m*(d-1));
end
