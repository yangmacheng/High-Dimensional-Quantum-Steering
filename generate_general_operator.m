%% 广义量子变换生成函数 (包含幺正与反幺正)
function [U, is_anti] = generate_general_operator(d, params)
    % 输入 params 的长度应当为 d^2 + 1
    % params(1:end-1)用于生成 U
    % params(end)    用于决定性质 (Unitary vs Anti-unitary)
    
    % 1. 判定变换类型
    % 使用最后一个参数作为“开关”。
    % 在CEM优化中，如果初始均值为0，方差较大，它会自动探索两类变换。
    switch_param = params(end);
    if switch_param > 0
        is_anti = true;  % 反幺正 (Anti-unitary)
    else
        is_anti = false; % 幺正 (Unitary)
    end
    
    % 2. 提取用于生成矩阵 U 的参数
    u_params = params(1:end-1);
    
    % 3. 生成酉矩阵 U (复用之前的 U(d) 逻辑)
    H = zeros(d);
    idx = 1;
    
    % 对角元素
    for i = 1:d
        H(i, i) = u_params(idx);
        idx = idx + 1;
    end
    
    % 非对角元素
    for i = 1:d
        for j = (i+1):d
            real_part = u_params(idx);
            imag_part = u_params(idx+1);
            H(i, j) = real_part + 1i * imag_part;
            H(j, i) = real_part - 1i * imag_part;
            idx = idx + 2;
        end
    end
    
    % 数值保护与指数映射
    max_norm = max(abs(eig(H)));
    if max_norm > 5
        H = H * 5 / max_norm;
    end
    U = expm(1i * H);
end
