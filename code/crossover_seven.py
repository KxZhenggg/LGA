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

import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.calculator import SevenNetCalculator



class SevenNet_Latent_Opt_Calculator(SevenNetCalculator):
    def __init__(
        self,
        latent_target: torch.Tensor = None,
        layer_name: str = "11_equivariant_gate",
        opt_coeff: float = 1.0,
        **kwargs
    ):

        super().__init__(**kwargs)
        
        self.layer_name = layer_name
        self.opt_coeff = opt_coeff
        self.latent_target = None
        
        if latent_target is not None:
            self.set_latent_target(latent_target)


    def set_latent_target(self, latent_target):
        if isinstance(latent_target, np.ndarray):
            latent_target = torch.from_numpy(latent_target)
        self.latent_target = latent_target.to(self.device).detach()


    def forward_latent_with_grad(self, atoms):
        """
        核心函数：构建图 -> 施加变形 -> Hook截取 -> 抛出异常中断
        """
        is_ts_type = isinstance(self.model, torch.jit.ScriptModule)
        numpy_dict = unlabeled_atoms_to_graph(atoms, self.cutoff, with_shift=is_ts_type)
        data = AtomGraphData.from_numpy_dict(numpy_dict)
        
        if self.modal: data[KEY.DATA_MODALITY] = self.modal
        data.to(self.device)

        if KEY.POS not in data: raise RuntimeError(f"Key {KEY.POS} not found.")

        """strain_tensor= torch.zeros((3, 3), device=self.device, requires_grad=True)
        transformation = torch.eye(3, device=self.device) + strain_tensor
        
        data[KEY.POS].requires_grad_(True)
        data[KEY.POS] = torch.matmul(data[KEY.POS], transformation)
        pos_tensor = data[KEY.POS] """
        
    
        
        strain_leaf = torch.zeros((3, 3), device=self.device, requires_grad=True)
        transformation = torch.eye(3, device=self.device) + strain_leaf
        
        pos_leaf = data[KEY.POS].clone().detach().requires_grad_(True)
        pos_input = torch.matmul(pos_leaf, transformation)
        data[KEY.POS] = pos_input


        """if KEY.EDGE_VEC in data:
            edge_orig = data[KEY.EDGE_VEC].clone().detach()
            data[KEY.EDGE_VEC] = torch.matmul(edge_orig, transformation)
            
            if KEY.EDGE_LENGTH in data:
                data[KEY.EDGE_LENGTH] = torch.norm(data[KEY.EDGE_VEC], p=2, dim=-1, keepdim=True)
            else:
                pass"""
                
        if KEY.EDGE_VEC in data and KEY.EDGE_IDX in data:
            edge_index = data[KEY.EDGE_IDX] # (2, E)
            src, dst = edge_index[0], edge_index[1]
            
            edge_vec_orig = data[KEY.EDGE_VEC].detach()
            pos_orig = data[KEY.POS].detach() 
            pos_diff_orig = pos_orig[dst] - pos_orig[src]
            shifts = edge_vec_orig - pos_diff_orig
            
            
            shifts_input = torch.matmul(shifts, transformation)
            
            pos_diff_input = pos_input[dst] - pos_input[src]
            
            edge_vec_input = pos_diff_input + shifts_input
            
            data[KEY.EDGE_VEC] = edge_vec_input
            
            if KEY.EDGE_LENGTH in data:
                data[KEY.EDGE_LENGTH] = torch.norm(edge_vec_input, p=2, dim=-1, keepdim=True)
        
        else:
            print("Warning: Edge data missing, gradients might be wrong.")

        features_container = {}

        class StopForwardException(Exception):
            pass

        def hook_fn(module, input, output):

            if isinstance(output, (dict, AtomGraphData)):
                if KEY.NODE_FEATURE in output: features_container['feat'] = output[KEY.NODE_FEATURE]
            elif isinstance(output, torch.Tensor): features_container['feat'] = output
            elif isinstance(output, tuple): features_container['feat'] = output[0]

            raise StopForwardException()

        
        target_module = None
        if self.layer_name:
            modules = dict(self.model.named_modules())
            target_module = modules.get(self.layer_name)
        if target_module is None:
             layers = list(self.model.children())
             target_module = layers[-2] if len(layers) > 1 else layers[-1]

        handle = target_module.register_forward_hook(hook_fn)

        try:
            self.model(data)
        except StopForwardException:
            pass
        except Exception as e:
            raise e
        finally:
            handle.remove()

        latent_out = features_container.get('feat')
        if latent_out is None:
            raise RuntimeError("Hook failed to capture features. Check layer_name.")

        return pos_leaf, strain_leaf, latent_out


    def calculate(self, atoms=None, properties=None, system_changes=all_changes):

        self.results = {} # 强制清理

        Calculator.calculate(self, atoms, properties, system_changes)

        if self.latent_target is None: raise ValueError("Latent target not set.")

        pos_leaf, strain_leaf, latent_predict = self.forward_latent_with_grad(atoms)

        diff = torch.mean(latent_predict,dim=0) - self.latent_target
        loss = self.opt_coeff * torch.norm(diff, p=2)
        
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
           

    def get_descriptors(self, atoms):
        if atoms is None: atoms = self.atoms
        _, _, latent = self.forward_latent_with_grad(atoms)
        return latent.detach().cpu().numpy()


def main():
    
    calc = SevenNet_Latent_Opt_Calculator(
        model="checkpoint_sevennet_omni_i12.pth",
        modal='matpes_pbe',
        enable_cueq=False,
        enable_flash=False,
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
