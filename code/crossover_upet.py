import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from ase.io import read, write
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.filters import FrechetCellFilter
from ase.neighborlist import neighbor_list

from upet.calculator import UPETCalculator
from metatomic.torch import System, ModelEvaluationOptions, ModelOutput, NeighborListOptions
from metatensor.torch import Labels, TensorBlock


class UPET_Latent_Opt_Calculator(UPETCalculator):
    def __init__(
        self,
        latent_target: torch.Tensor = None,
        layer_name: str = "module.gnn_layers.4", 
        opt_coeff: float = 1.0,
        cutoff: float = 10.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not hasattr(self, "raw_model"):
            raise RuntimeError("UPETCalculator must store 'self.raw_model'. Please verify the parent class modification.")

        self.layer_name = layer_name
        self.opt_coeff = opt_coeff
        self.cutoff = cutoff
        self.latent_target = None
        
        try:
            param = next(self.raw_model.parameters())
            self.device = param.device
            self.dtype = param.dtype
        except:
            self.device = 'cpu'
            self.dtype = torch.float64 

        if latent_target is not None:
            self.set_latent_target(latent_target)

    def set_latent_target(self, latent_target):
        if isinstance(latent_target, np.ndarray):
            latent_target = torch.from_numpy(latent_target)
        self.latent_target = latent_target.to(self.device).detach()

    def set_opt_coeff(self, opt_coeff):
        self.opt_coeff = opt_coeff

    def forward_latent_with_grad(self, atoms):
        
        i_idx, j_idx, S_shift, _ = neighbor_list('ijSD', atoms, self.cutoff)
        
        i_idx_dev = torch.tensor(i_idx, dtype=torch.long, device=self.device)
        j_idx_dev = torch.tensor(j_idx, dtype=torch.long, device=self.device)
        S_shift_dev = torch.tensor(S_shift, dtype=self.dtype, device=self.device)
        
        types = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=self.device)

        pos_array = atoms.get_positions()
        pos_leaf = torch.tensor(pos_array, dtype=self.dtype, device=self.device, requires_grad=True)

        strain_leaf = torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
        
        cell_array = atoms.get_cell().array
        if cell_array.shape == (3,): cell_array = np.diag(cell_array)
        cell_orig = torch.tensor(cell_array, dtype=self.dtype, device=self.device) 
        
        transformation = torch.eye(3, dtype=self.dtype, device=self.device) + strain_leaf
        
        pos_input = torch.matmul(pos_leaf, transformation)

        cell_input = torch.matmul(cell_orig, transformation)
        
        r_i = pos_input[i_idx_dev]
        r_j = pos_input[j_idx_dev]

        shift_vecs = torch.matmul(S_shift_dev, cell_input)
        
        edge_vecs = (r_j - r_i + shift_vecs).unsqueeze(-1)

        system = System(types=types, positions=pos_input, cell=cell_input, pbc=pbc)
        
        samples_values = torch.stack([
            i_idx_dev.to(torch.int32),
            j_idx_dev.to(torch.int32),
            torch.tensor(S_shift[:, 0], dtype=torch.int32, device=self.device),
            torch.tensor(S_shift[:, 1], dtype=torch.int32, device=self.device),
            torch.tensor(S_shift[:, 2], dtype=torch.int32, device=self.device)
        ], dim=1)
        
        samples = Labels(
            names=["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            values=samples_values
        )

        components = [Labels(
            names=["xyz"], 
            values=torch.tensor([[0], [1], [2]], dtype=torch.int32, device=self.device)
        )]

        properties = Labels(
            names=["distance"], 
            values=torch.tensor([[0]], dtype=torch.int32, device=self.device)
        )
        
        nl_block = TensorBlock(
            values=edge_vecs,
            samples=samples,
            components=components,
            properties=properties
        )

        nl_options = NeighborListOptions(cutoff=self.cutoff, full_list=True, strict=True)
        system.add_neighbor_list(nl_options, nl_block)
        
        features_container = {}
        class StopForwardException(Exception): pass

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                features_container['feat'] = output
            elif isinstance(output, tuple) and len(output) > 0:
                features_container['feat'] = output[0]
            raise StopForwardException()

        target_module = None
        if self.layer_name:
            modules = dict(self.raw_model.named_modules())
            target_module = modules.get(self.layer_name)
        
        if target_module is None:
            for name in ["module.gnn_layers.4", "module.pet.body", "module.gnn"]:
                modules = dict(self.raw_model.named_modules())
                if name in modules:
                    target_module = modules[name]
                    break
        
        if target_module is None:
             raise ValueError(f"Could not find layer '{self.layer_name}' or default layers.")

        handle = target_module.register_forward_hook(hook_fn)

        eval_options = ModelEvaluationOptions(outputs={}, selected_atoms=None)

        try:
            self.raw_model([system], eval_options, check_consistency=False)
        except StopForwardException:
            pass
        except Exception as e:
            raise e
        finally:
            handle.remove()

        latent_out = features_container.get('feat')
        if latent_out is None:
            raise RuntimeError("Hook failed to capture features.")

        return pos_leaf, strain_leaf, latent_out


    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.results = {}

        Calculator.calculate(self, atoms, properties, system_changes)
        
        if self.latent_target is None:
            raise ValueError("Latent target not set.")

        pos_leaf, strain_leaf, latent_predict = self.forward_latent_with_grad(atoms)

        latent_pooled = torch.mean(latent_predict, dim=0)
        
        diff = latent_pooled - self.latent_target
        loss = torch.norm(diff, p=2)

        self.results['energy'] = loss.detach().cpu().item()
        self.results['free_energy'] = self.results['energy']

        grads = torch.autograd.grad(
            outputs=[loss],
            inputs=[pos_leaf, strain_leaf],
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        forces_grad, stress_grad = grads[0], grads[1]

        if forces_grad is not None:
            self.results['forces'] = -1 * forces_grad.detach().cpu().numpy()
        else:
            self.results['forces'] = np.zeros_like(atoms.positions)

        if stress_grad is not None:
            virial = 1.0 * stress_grad.detach().cpu().numpy()
            volume = atoms.get_volume()
            if atoms.pbc.any():
                stress_full = virial / volume
                self.results['stress'] = full_3x3_to_voigt_6_stress(stress_full)
            else:
                self.results['stress'] = np.zeros(6)
        else:
            self.results['stress'] = np.zeros(6)

    def get_descriptors(self, atoms=None):
        if atoms is None: atoms = self.atoms
        _, _, latent = self.forward_latent_with_grad(atoms)
        return latent.detach().cpu().numpy()


def main():
    
    calc = UPET_Latent_Opt_Calculator(
        model="pet-oam-xl", 
        checkpoint_path="upet.ckpt", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        opt_coeff=1.0
    )
    
    parent1 = read(f"../example/parent1.vasp", format="vasp")
    parent2 = read(f"../example/parent2.vasp", format="vasp")
    latent_1 = torch.tensor(np.mean(calc.get_descriptors(parent1), axis=0))
    latent_2 = torch.tensor(np.mean(calc.get_descriptors(parent2), axis=0))
    v_target = (0.5*latent_1+0.5*latent_2) 
    
    child_unrelaxed = read(f"../example/child_unrelaxed.vasp", format="vasp")
    
    calc.set_latent_target(v_target)
    child_unrelaxed.calc = calc
        
    opt = FIRE(FrechetCellFilter(child_unrelaxed))

    opt.run(fmax=0.00, steps=300)
    write(f"../example/child_relaxed", child_unrelaxed, format="vasp")


if __name__ == "__main__":
    main()
