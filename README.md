# Code for "*[Witness High-Dimensional Quantum Steering via Majorization Lattice](https://arxiv.org/abs/2507.20950)*"
### Ma-Cheng Yang and Cong-Feng Qiao

This is a repository for code which was written for the article "*Witness High-Dimensional Quantum Steering via Majorization Lattice*. Ma-Cheng Yang and Cong-Feng Qiao. [arXiv:2507.20950 [quant-ph]](https://arxiv.org/abs/2507.20950)."

All codes are written in MATLAB and requires *Parallel Computing Toolbox*.

The code mainly contains two parts: solving Cross-Entropy optimization for the optimal settings of qutrit Werner and isotropic states, and steering detection for general scenario.
  
#### Cross-Entropy optimization for the optimal settings of qutrit Werner and isotropic states

- **[`mub2to8.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/mub2to8.mat)**
  - Datas of mutually unbiased bases with dimension $2$ to $8$

- **[`omegak_batching_optimized_bloch()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/omegak_batching_optimized_Bloch.m)**
  - Calculate $\Omega_k$, Bloch parametrization employing the cross-entropy optimization

- **[`crossEntropyOptimizer()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/crossEntropyOptimizer.m)**
  - Find the optimal measurement settings for $N$-measurement scenario for Werner and isotropic states
 
- **[`isotropic_werner_cem_para_script.mlx`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/isotropic_werner_cem_para_script.mlx)**
  - The script of parameters setting of cross-entropy optimization for Werner and isotropic states

#### Steering detection for general scenario

- **[`omegak_batching_optimized()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/omegak_batching_optimized.m)**
  - Calculate $\Omega_k$
 
- **[`base_array_general_operation()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/base_array_general_operation.m)**
  - Perform a generalized transformation (unitary and anti-unitary) on the base vectors

- **[`calc_quantum_prob_aggregatio()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/calc_quantum_prob_aggregation.m)**
  - Calculate quanutm measurement probability distributions after aggregation for a given measurement bases
 
- **[`calc_shannon_entropy()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/calc_shannon_entropy.m)**
  - Calculate Shannon entropy

- **[`generate_general_operator()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/generate_general_operator.m)**
  - Generator general transformation (unitary and anti-unitary) in light of the given parametres

- **[`minimization_entropy()`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/minimization_entropy.m)**
  - Find the optimal alignment between Alice's measurement and Bob's one for general quantum states and measurements.

- **[`general_case_script.mlx`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/general_case_script.mlx)**
  - The script of parameters setting of cross-entropy optimization for general scenario
