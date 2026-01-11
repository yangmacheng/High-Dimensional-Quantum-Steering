function base_array_after_operation = base_array_general_operation(base_array, general_operator)
% BASE_ARRAY_GENERAL_OPERATION 对 m 组 d 维基矢量做广义变换
% 输入:
%   base_array: 3D 数组 [d, d, M], 表示 m 组 d 维测量基矢量，其中 base_array(:, :, i) 每行表示第
%   i 组基的测量基矢量
%   general_operator: 结构体
%       .U       : 变换矩阵 (d x d)
%       .is_anti : 布尔值 (true/false)

    % 1. 解包结构体
    U = general_operator.U;
    is_anti = general_operator.is_anti;

    % 2. 反幺正处理
    % 物理含义: 反幺正 = 复共轭 + 幺正旋转
    if is_anti
        base_array = conj(base_array);
    end

    % 3. 页矩阵乘法 (核心优化)
    % -----------------------------------------------------------
    % 语法含义: pagemtimes(A, transA, B, transB)
    % A = base_array, 操作 = 'none' (不转置)
    % B = U,          操作 = 'transpose' (非共轭转置 U.')
    % -----------------------------------------------------------
    % 结果等效于对每一页 k 执行: 
    % Result(:,:,k) = base_array(:,:,k) * U.'
    
    base_array_after_operation = pagemtimes(base_array, 'none', U, 'transpose');

end
