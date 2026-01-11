# Code for "Witness High-Dimensional Quantum Steering via Majorization Lattice"

[![arXiv](https://img.shields.io/badge/arXiv-2507.20950-b31b1b.svg)](https://arxiv.org/abs/2507.20950)
[![Language](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)

**Authors:** Ma-Cheng Yang and Cong-Feng Qiao

This repository contains the source code associated with the research article:
> **Witness High-Dimensional Quantum Steering via Majorization Lattice**  
> Ma-Cheng Yang and Cong-Feng Qiao  
> *arXiv preprint arXiv:2507.20950 [quant-ph]* (2025).  
> [View on arXiv](https://arxiv.org/abs/2507.20950)

## 📋 Requirements

The codebase is developed in **MATLAB**. To run the parallel optimization scripts efficiently, the following toolbox is required:

- **MATLAB Parallel Computing Toolbox**

## 📂 Repository Structure

The code consists of two main modules:

- **Solving Cross-Entropy Method (CEM) optimization for optimal settings in qutrit Werner and isotropic states**

- **General steering detection for arbitrary quantum states and measurements**

---

### Cross-Entropy Optimization (Werner & Isotropic States)
*Focuses on finding optimal measurement settings for N-measurement scenarios.*

- **[`mub2to8.mat`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/mub2to8.mat)**
  - Mutually Unbiased Bases (MUBs) in dimensions $d=2$ to $8$.

- **[`crossEntropyOptimizer.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/crossEntropyOptimizer.m)**
  - **Core Algorithm**: Finds the optimal measurement settings for Werner and isotropic states using the Cross-Entropy Method.

- **[`omegak_batching_optimized_Bloch.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/omegak_batching_optimized_Bloch.m)**
  - Computes $\Omega_k$ using Bloch parametrization optimized via cross-entropy.

- **[`isotropic_werner_cem_para_script.mlx`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/isotropic_werner_cem_para_script.mlx)**
  - **Live Script**: Configuration and parameter settings for running the optimization on Werner/isotropic states.


### General Steering Detection
*Algorithms for minimizing entropy and detecting steering in general scenarios.*

- **[`minimization_entropy.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/minimization_entropy.m)**
  - **Core Algorithm**: Optimizes the alignment between Alice's and Bob's measurements (Unitary/Anti-unitary) to minimize entropy.

- **[`omegak_batching_optimized.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/omegak_batching_optimized.m)**
  - Calculates the majorization bound $\Omega_k$.

- **[`calc_quantum_prob_aggregation.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/calc_quantum_prob_aggregation.m)**
  - Computes the aggregated quantum measurement probability distributions for given measurement bases.

- **[`calc_shannon_entropy.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/calc_shannon_entropy.m)**
  - Helper function to calculate Shannon entropy.

- **[`base_array_general_operation.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/base_array_general_operation.m)**
  - Applies generalized transformations (Unitary and Anti-unitary) to the basis vectors.

- **[`generate_general_operator.m`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/generate_general_operator.m)**
  - Generates specific unitary or anti-unitary operators based on input parameters.

- **[`general_case_script.mlx`](https://github.com/yangmacheng/High-Dimensional-Quantum-Steering/blob/main/general_case_script.mlx)**
  - **Live Script**: Main execution script for the general scenario simulation.

##  citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{yang2025witness,
  title={Witness High-Dimensional Quantum Steering via Majorization Lattice},
  author={Yang, Ma-Cheng and Qiao, Cong-Feng},
  journal={arXiv preprint arXiv:2507.20950},
  year={2025}
}
