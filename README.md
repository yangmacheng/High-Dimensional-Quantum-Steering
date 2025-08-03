# Code for "*[Witness the High-Dimensional Quantum Steering via Majorization Lattice](https://arxiv.org/abs/2507.20950)*"
#### Ma-Cheng Yang and Cong-Feng Qiao

This is a repository for code which was written for the article "*Witness the High-Dimensional Quantum Steering via Majorization Lattice*. Ma-Cheng Yang and Cong-Feng Qiao. [arXiv:2507.20950 [quant-ph]](https://arxiv.org/abs/2507.20950)."

All codes are written in MATLAB and requires *Parallel Computing Toolbox*

The code mainly contains two parts: solving $\Omega_k$ and Cross-Entropy optimization

- Solving $\Omega_k$
  
The calculation of $\Omega_k$ is a typical combinatorial optimization problem (COP), where we can employ the MATLAB parallel toolbox "*Parallel Computing Toolbox*" to solve $\Omega_k$  for certain measurement settings via omegak_batching_optimized.m (measurement bases) or omegak_batching_optimized_Bloch.m (Bloch parametrization).

- Cross-Entropy optimization

Find the optimal measurement settings for $N$-measurement scenario, where qutrit isotropic and Werner states correspond to crossEntropyOptimizerIsotropic.m and crossEntropyOptimizerWerner.m, respectively.

