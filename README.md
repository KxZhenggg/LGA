# LGA Supplementary Files

This repository contains supplementary data and scripts for the latent genetic algorithm (LGA) workflow reported in the following article:

```bibtex
@misc{zheng2026latentgeneticalgorithmcrystal,
      title={Latent Genetic Algorithm for Crystal Structure Prediction},
      author={Kaixin Zheng and Wanjian Yin and Hongyu Yu and Hongjun Xiang},
      year={2026},
      eprint={2606.29220},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2606.29220},
}
```

## Data

The `data/` directory contains partial structural information for the superlattice sections investigated in this work. The files are provided in POSCAR-style format and include representative compact and long-period PTO/PZO superlattice configurations.

## Code

The `code/` directory contains the main LGA crossover scripts. The four script variants correspond to the following interatomic potential frameworks and model files:

| Script | Framework version | Potential model |
| --- | --- | --- |
| `crossover_mace.py` | MACE 0.3.13 | `2023-12-10-mace-128-L0_energy_epoch-249.model` |
| `crossover_matsim.py` | MatterSim 1.2.0 | `mattersim-v1.0.0-5M.pth` |
| `crossover_seven.py` | SevenNet 0.12.0 | `checkpoint_sevennet_omni_i12.pth` |
| `crossover_upet.py` | UPET 0.1.1 | `pet-oam-xl-v1.0.0.ckpt` |

These scripts provide the core crossover implementation used in the LGA workflow. Before applying them to a specific system, users should modify the structural information, model paths, and the relevant interatomic-potential source code according to the actual material system and computational setup.
