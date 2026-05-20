import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import time

from ase.units import GPa
from ase.io import read, write
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.filters import FrechetCellFilter

from mattersim.forcefield.potential import Potential, batch_to_dict
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield import MatterSimCalculator



class MatterSim_Latent_Opt_Calculator(Calculator):

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        potential: Potential = None,
        load_path: str = None, 
        latent_target: torch.Tensor = None, 
        layer_name: str = "graph_conv.3", 
        opt_coeff: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        args_dict: dict = {},
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        
        self.device = device
        self.args_dict = args_dict
        self.layer_name = layer_name
        self.opt_coeff = opt_coeff
        self.latent_target = None

        if potential is None:
            print(f"Loading MatterSim model from: {load_path if load_path else 'default m3gnet'}...")
            self.potential = Potential.from_checkpoint(
                load_path=load_path, 
                device=device, 
                **kwargs
            )
        else:
            self.potential = potential

        self.potential.model.eval()

        if latent_target is not None:
            self.set_latent_target(latent_target)

    def set_latent_target(self, latent_target):
        if isinstance(latent_target, np.ndarray):
            latent_target = torch.from_numpy(latent_target)
        self.latent_target = latent_target.to(self.device).detach()

    def set_opt_coeff(self, opt_coeff):
        self.opt_coeff = opt_coeff

    def forward_latent_with_grad(self, atoms):
        
        calc_args = self.args_dict.copy()
        calc_args["batch_size"] = 1
        calc_args["only_inference"] = True
        
        cutoff = self.potential.model.model_args.get("cutoff", 5.0)
        threebody_cutoff = self.potential.model.model_args.get("threebody_cutoff", 4.0)

        dataloader = build_dataloader(
            [atoms],
            model_type=self.potential.model_name,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            **calc_args,
        )
        
        graph_batch = next(iter(dataloader))
        input_data = batch_to_dict(graph_batch, model_type=self.potential.model_name)
        
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                input_data[k] = v.to(self.device)

        strain_tensor = torch.zeros((3, 3), device=self.device, requires_grad=True)
        
        transformation = torch.eye(3, device=self.device) + strain_tensor

        if "atom_pos" in input_data:
            pos_tensor = input_data["atom_pos"].clone().detach().requires_grad_(True)
            pos_input = torch.matmul(pos_tensor, transformation)
            input_data["atom_pos"] = pos_input
            
        else:
            raise RuntimeError("input dictionary is missing 'atom_pos'.")

        if "cell" in input_data:
            input_data["cell"] = torch.matmul(input_data["cell"], transformation)
        else:
            pass

        features_container = {}
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple): target_tensor = output[0]
                else: target_tensor = output
                features_container[name] = target_tensor
            return hook

        modules = dict(self.potential.model.named_modules())
        if self.layer_name not in modules:
            raise ValueError(f"Layer '{self.layer_name}' not found in model.")
            
        target_module = modules[self.layer_name]
        handle = target_module.register_forward_hook(get_activation("latent"))

        try:
            self.potential.model(input_data) 
            latent_out = features_container.get("latent")
            if latent_out is None: raise RuntimeError("Hook failed to capture features.")
        finally:
            handle.remove()

        return pos_tensor, strain_tensor, latent_out


    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        
        if self.latent_target is None:
            raise ValueError("Latent target not set.")

        pos_tensor, strain_tensor, latent_predict = self.forward_latent_with_grad(atoms)
        
        diff = torch.mean(latent_predict, dim=0) - self.latent_target
        loss = torch.norm(diff, p=2) * self.opt_coeff
        
        self.results = {}
        self.results["energy"] = loss.detach().cpu().item()
        self.results["free_energy"] = self.results["energy"]

        grads = torch.autograd.grad(
            outputs=[loss],
            inputs=[pos_tensor, strain_tensor],
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )
        
        forces_grad = grads[0]
        stress_grad = grads[1] 
        
        if forces_grad is not None:
            self.results["forces"] = -1 * forces_grad.detach().cpu().numpy()
        else:
            self.results["forces"] = np.zeros_like(atoms.positions)

        if stress_grad is not None:
            virial = 1 * stress_grad.detach().cpu().numpy()
            volume = atoms.get_volume()
            
            if atoms.pbc.any():
                stress_full = virial / volume
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress_full)
            else:
                self.results["stress"] = np.zeros(6)
        else:
            self.results["stress"] = np.zeros(6)
            
    def get_descriptors(self, atoms):
        _, _, des = self.forward_latent_with_grad(atoms)
        return des.detach().cpu().numpy()

def main():
    
    calc = MatterSim_Latent_Opt_Calculator(
        load_path="mattersim.pth", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        layer_name="graph_conv.3",
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
