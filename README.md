# Code for "*[Witness High-Dimensional Quantum Steering via Majorization Lattice](https://arxiv.org/abs/2507.20950)*"
#### Ma-Cheng Yang and Cong-Feng Qiao

This is a repository for code which was written for the article "*Witness High-Dimensional Quantum Steering via Majorization Lattice*. Ma-Cheng Yang and Cong-Feng Qiao. [arXiv:2507.20950 [quant-ph]](https://arxiv.org/abs/2507.20950)."

All codes are written in MATLAB and requires *Parallel Computing Toolbox*.

The code mainly contains three parts: solving $\Omega_k$, Cross-Entropy optimization for the optimal settings of qutrit Werner and isotropic states, and steering detection for general scenario.

- Solving $\Omega_k$
  
The calculation of $\Omega_k$ is a typical combinatorial optimization problem (COP), where we can employ the MATLAB parallel toolbox "*Parallel Computing Toolbox*" to solve $\Omega_k$  for certain measurement settings via omegak_batching_optimized.m (measurement bases) or omegak_batching_optimized_Bloch.m (Bloch parametrization).

-  Cross-Entropy optimization for the optimal settings of qutrit Werner and isotropic states

Try to find the optimal measurement settings for $N$-measurement scenario (crossEntropyOptimizer.m).

- Steering for general scenario (here, we provide a qutrit example) including 

omegak_batching_optimized.m:

### 核心功能

- **[`calculate_steering()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/omegak_batching_optimized.m)**
  - 用于计算高维量子导引的核心指标。
  - 使用了 SDP 半定规划算法。

- **[`plot_result()`](https://github.com/your/repo/blob/main/code.py#L45)**
  - **功能**：将计算结果可视化。
  - **注意**：需要安装 `matplotlib` 库。
