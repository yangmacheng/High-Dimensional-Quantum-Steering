classdef CombGenerator < handle
% 组合生成器类 - 高效生成字典序组合
% 使用基于格雷码的迭代方法，避免索引计算错误

properties (Access = private)
    n        % 总元素数
    k        % 组合大小
    current_comb  % 当前组合
    total_comb   % 总组合数
    current_index % 当前组合索引
end

methods
    function obj = CombGenerator(n, k, start_index, total_comb)
        % 构造函数：初始化生成器
        obj.n = n;
        obj.k = k;
        obj.total_comb = total_comb;
        obj.current_index = 0;  % 从0开始计数
        
        % 验证起始索引
        if start_index < 1 || start_index > total_comb
            error('起始索引超出范围: %d (有效范围: 1-%d)', start_index, total_comb);
        end
        
        % 初始化组合
        obj.current_comb = obj.initialize_comb();
        
        % 前进到起始索引
        for i = 1:start_index-1
            obj.getNext();
        end
    end
    
    function comb = getNext(obj)
        % 获取下一个组合（字典序）
        if obj.current_index >= obj.total_comb
            error('已超过最大组合数: %d', obj.total_comb);
        end
        
        % 如果是第一个组合，直接返回
        if obj.current_index == 0
            obj.current_index = 1;
            comb = obj.current_comb;
            return;
        end
        
        % 寻找最右侧可递增的位置
        i = obj.k;
        while i >= 1
            % 如果当前数字可以递增
            if obj.current_comb(i) < obj.n - obj.k + i
                obj.current_comb(i) = obj.current_comb(i) + 1;
                
                % 重置右侧所有数字
                for j = i+1:obj.k
                    obj.current_comb(j) = obj.current_comb(j-1) + 1;
                end
                
                obj.current_index = obj.current_index + 1;
                comb = obj.current_comb;
                return;
            end
            i = i - 1;
        end
        
        error('无法生成下一个组合');
    end
end

methods (Access = private)
    function comb = initialize_comb(obj)
        % 初始化第一个组合 [1, 2, ..., k]
        comb = 1:obj.k;
    end
end
end
