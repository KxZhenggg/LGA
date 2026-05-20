# Latent Genetic Algorithm (LGA)

This repository contains the model-specific crossover scripts and structural data associated with the manuscript:

**Latent-space genetic algorithms for efficient crystal structure prediction**

The code implements latent-space crossover using four pretrained universal machine-learning interatomic potentials: MACE, MatterSim, SevenNet, and UPET. The repository also provides selected VASP-relaxed superlattice structures in POSCAR format.

---

## Repository structure

```text
.
├── code/
│   ├── crossover_mace.py
│   ├── crossover_mattersim.py
│   ├── crossover_sevennet.py
│   └── crossover_upet.py
├── data/
│   ├── POSCAR_PTO_PZO_n1_compact
│   ├── POSCAR_PTO_PZO_n2_compact
│   ├── POSCAR_PTO_PZO_n3_compact
│   ├── POSCAR_PTO_PZO_n4_compact
│   ├── POSCAR_PTO_PZO_n4_long_period
│   ├── POSCAR_PTO_PZO_n5_compact
│   ├── POSCAR_PTO_PZO_n5_long_period
│   ├── POSCAR_BTO_BZO_n4_SM2_0_C2_0
│   └── POSCAR_BTO_BZO_n4_SM2_1_C2_1
├── examples/
│   ├── parent1.vasp
│   ├── parent2.vasp
│   ├── child_unrelaxed.vasp
│   └── child_relaxed.vasp
├── LICENSE
├── DATA_LICENSE.md
└── README.md
```

---

## Code

The `code/` directory provides four model-specific LGA crossover scripts:

```text
crossover_mace.py
crossover_mattersim.py
crossover_sevennet.py
crossover_upet.py
```

Each script reads two parent structures, extracts model-specific atomic descriptors, constructs a target latent representation by weighted interpolation of the parent latent vectors, and performs inverse optimization of atomic positions and lattice degrees of freedom to generate an offspring structure.

To run an example:

```bash
python code/crossover_mace.py
python code/crossover_mattersim.py
python code/crossover_sevennet.py
python code/crossover_upet.py
```

Users may need to modify local paths to parent structures, checkpoints, and output files.

---

## Model and descriptor details

| Model | Software version | Checkpoint | Descriptor used for LGA |
|---|---:|---|---|
| MACE | 0.3.13 | `2023-12-10-mace-128-L0_epoch-199.model` | `node_feats` from the forward pass; invariant components are extracted before atom-wise mean pooling |
| MatterSim | 1.2.0 | `mattersim-v1.0.0-5M.pth` | `graph_conv.3` hidden atomic features; mean-pooled over atoms |
| SevenNet | 0.12.0 | `checkpoint_sevennet_omni_i12.pth` | `11_equivariant_gate` hidden atomic features; mean-pooled over atoms |
| UPET | 0.1.1 | `pet-oam-xl-v1.0.0.ckpt` | `module.gnn_layers.4` hidden atomic features; mean-pooled over atoms |

The checkpoint names above specify the model versions used in the manuscript. Pretrained model checkpoints are subject to the licenses and distribution terms of their original providers. If checkpoint files are not included in this repository, users should obtain them from the corresponding official sources.

---

## Structural data

The `data/` directory contains selected VASP-relaxed superlattice structures in POSCAR format.

The `PTO_PZO` files correspond to compact and long-period structures of the \((\mathrm{PbTiO}_3)_n/(\mathrm{PbZrO}_3)_n\) superlattices discussed in the manuscript.

The `BTO_BZO` files correspond to the \((\mathrm{BaTiO}_3)_4/(\mathrm{BaZrO}_3)_4\) control structures used in the frozen-mode analysis.

---

## License

The source code in this repository is licensed under the MIT License; see `LICENSE`.

The structural data files in the `data/` directory are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0); see `DATA_LICENSE.md`.

Third-party packages, pretrained models, and checkpoints, including MACE, MatterSim, SevenNet, UPET, PASP, ASE, and VASP, are subject to their own respective licenses and are not covered by the licenses of this repository.

---

## Citation

If you use this code or structural data, please cite the associated manuscript:

```text
K. Zheng, W. Yin, H. Yu, and H. Xiang,
Latent-space genetic algorithms for efficient crystal structure prediction.
```

A formal citation will be added after publication.