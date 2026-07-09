import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from typing import Union, Optional

from ase import Atoms, Atom
from ase.io import read, write, Trajectory
from ase.optimize import BFGS, FIRE
from ase.build.supercells import make_supercell
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.filters import FrechetCellFilter, ExpCellFilter, UnitCellFilter

from mace import data
from mace.data.atomic_data import AtomicData
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric
from mace.calculators import MACECalculator



class MACE_Latent_Opt_Calculator(MACECalculator):
    """MACE ASE Calculator
       for latent vector optimization
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        charges_key: str, Array field of atoms object where atomic charges are stored
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]

    Dipoles are returned in units of Debye
    """

    def __init__(
        self,
        model_paths: Union[list, str],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        latent_target: Union[None, torch.Tensor] = None,
        opt_coeff: float = 1.0,
        **kwargs,
    ):
        MACECalculator.__init__(
            self,
            model_paths=model_paths,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            model_type=model_type,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            **kwargs,
        )
        self.latent_target = latent_target
        self.opt_coeff = opt_coeff

    
    def set_latent_target(self, latent_target):
        self.latent_target = latent_target


    def forward_model(self, atomic_data:AtomicData, invariants_only=True, num_layers=-1):
        """
        preform forward computation 
        return parameters for gradient calculation
        """

        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        if num_layers == -1:
            num_layers = int(self.models[0].num_interactions)

        data_loader = torch_geometric.dataloader.DataLoader(
            
            dataset=[atomic_data],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        data = next(iter(data_loader)).to(self.device)
        out = self.models[0](
            data.to_dict(),
            training = True,
            compute_displacement = True
        )

        pos = data["positions"]
        displacement = out["displacement"]
        descriptor = out["node_feats"]
        
        if invariants_only:
            irreps_out = self.models[0].products[0].linear.__dict__["irreps_out"]
            l_max = irreps_out.lmax
            num_features = irreps_out.dim // (l_max + 1) ** 2
            descriptor = extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_features,
                    l_max=l_max,
                )
    
        return pos, displacement, torch.mean(descriptor, dim=0)


    def backward_latent(self, atoms):
        """
        use parameters from forward_latent()
        calculate energy, forces, virials and stress
        """
        config = data.config_from_atoms(atoms)
        atomic_data = AtomicData.from_config(
            config, 
            z_table=self.z_table, 
            cutoff=self.r_max
        )
        atomic_data = atomic_data.to(self.device)

        positions, disreplacement, latent_predict = self.forward_model(atomic_data)
        latent_target = self.latent_target.detach().to(self.device)   
        
        energy = torch.sqrt(torch.sum((latent_predict - latent_target) ** 2)) \
                 * self.opt_coeff

        atomic_data["energy"] = energy

        forces: Optional[torch.Tensor] = torch.zeros_like(atomic_data["positions"])
        virials: Optional[torch.Tensor] = torch.zeros_like(atomic_data["cell"])

        forces, virials = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions, disreplacement],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        if forces is None:
            assert False, "failed to compute force autograd!"
        atomic_data["forces"] = -1 * forces

        if virials is None:
            assert False, "failed to compute vector_force autograd!"
        atomic_data["virials"] = -1 * virials

        cell = atomic_data["cell"].view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        atomic_data["stress"] = stress

        return atomic_data


    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        Calculator.calculate(self, atoms)

        out = self.backward_latent(atoms)

        self.results = {}

        self.results["energy"] = (
                out["energy"]
                .detach()
                .cpu()
                .item()
                * self.energy_units_to_eV
            )
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = (
                out["forces"]
                .detach()
                .cpu()
                .numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
        self.results["stress"] = full_3x3_to_voigt_6_stress(
                torch.mean(out["stress"], dim=0)
                .detach()
                .cpu()
                .numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A**3
            )


def main():
    
    calc = MACE_Latent_Opt_Calculator(
        model_paths="_mace.model", 
        device="cuda", 
        opt_coeff=1.0
    )
    
    parent1 = read(f"parent1.vasp", format="vasp")
    parent2 = read(f"parent2.vasp", format="vasp")
    latent_1 = torch.tensor(np.mean(calc.get_descriptors(parent1), axis=0))
    latent_2 = torch.tensor(np.mean(calc.get_descriptors(parent2), axis=0))
    v_target = (0.5*latent_1+0.5*latent_2) 
    
    child_unrelaxed = read(f"child_unrelaxed.vasp", format="vasp")
    
    calc.set_latent_target(v_target)
    child_unrelaxed.calc = calc
        
    opt = FIRE(FrechetCellFilter(child_unrelaxed))
    opt.run(fmax=0.00, steps=300)
    write("child_relaxed.vasp", child_unrelaxed, format="vasp")

if __name__ == "__main__":
    main()
