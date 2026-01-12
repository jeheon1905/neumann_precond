import itertools
import torch
import numpy as np

from gospel.scf import SCF
from gospel.Eigensolver import parallel_orthonormalize, BaseEigensolver
from gospel.ParallelHelper import ParallelHelper as PH

dot = lambda x,y : torch.sum(x*y, dim=0)

class LBFGS(SCF):
    def __init__(self, convg, occupation, param, density ):
        self.param = param

        self.maxiter = convg.get("scf_maxiter")
        self.energy_tol = convg.get("energy_tol")
        self.density_tol = convg.get("density_tol")

        self.occupation = occupation
        self.density = density

        self.eigvec = None
        self.eigval = None
        self.energy = None
        self.fermi_level = None

        assert self.occupation.temperature==0.0, 'LBFGS process in GOSPEL program ignores electronic temperature'
        # l-bfgs related variables
        self.s_list = []
        self.y_list = []
        self.rho_list = []


        self.converged = False
        self.rho_prev = None    # This rho is different to rho_list, rho_prev contains density values in previous iteration
        self.density_diff = None
        self.total_energy_list = [torch.nan]

    def __str__(self):
        s = str()
        s += "\n============================= [ LBFGS ] =============================="
        s += f"\n* m (memory)         : {self.param['m']}"
        s += f"\n* energy_tol         : {self.energy_tol}"
        s += f"\n* density_tol        : {self.density_tol}"
        s += "\n=====================================================================\n"
        return s

    
    def __two_loop_recursion(self, grad, eps=1e-10):
        q = grad.clone()
        alpha = []
        for s, y, rho in zip(reversed(self.s_list), reversed(self.y_list), reversed(self.rho_list)):
            alpha_i = rho * dot(s, q)
            #alpha_i = rho * torch.dot(s.view(-1), q.view(-1))
            alpha.append(alpha_i)
            q = q - alpha_i * y

        # Scaling of initial Hessian approximation
        if len(self.y_list) > 0:
            gamma = dot(self.s_list[-1], self.y_list[-1]) / (dot(self.y_list[-1], self.y_list[-1])+eps)
            #gamma = torch.dot(self.s_list[-1].view(-1), self.y_list[-1].view(-1)) / torch.dot(self.y_list[-1].view(-1), self.y_list[-1].view(-1))
        else:
            gamma = torch.tensor(1.0)
        r = gamma * q

        for s, y, rho, alpha_i in zip(self.s_list, self.y_list, self.rho_list, reversed(alpha)):
            beta = rho * dot(y, r)
            #beta = rho * torch.dot(y.view(-1), r.view(-1))
            r = r + s * (alpha_i - beta)
        return r

    def __update(self, s, y):
        if (type(s) ==np.ndarray):
            s = torch.cat ([ _ for _ in s.reshape(-1) ], dim=-1 )

        if len(self.s_list) == self.param['m']:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)

        self.s_list.append(s)
        self.y_list.append(y)
        self.rho_list.append(1.0 / dot(y, s))
        #self.rho_list.append(1.0 / torch.dot(y.view(-1), s.view(-1)))

    def iterate(self, hamiltonian, print_energies=False):
        for i_iter in range(self.maxiter):
            print(
                f"\n=================== [ LBFGS CYCLE {i_iter + 1} ] ==================="
            )

            if i_iter ==0: ## Set initial guess orbitals
                init_density = self.density.init_density()
                self.density.set_density(init_density)
                hamiltonian.update(self.density)

                # eigval is arbitrary sorted number no physical meaning!
                eigval = torch.arange(hamiltonian.nbands, 
                                      device = hamiltonian.device,
                                      dtype = init_density.dtype ).repeat(hamiltonian.nspins,hamiltonian.nibzkpts,1)

                # occupation is not changed during iteration
                # becuase orbital energies are not evaluated
                occ = self.occupation.get_occupation( eigval )

                _, orbital = BaseEigensolver.initialize_guess(hamiltonian)

                energy = energy_function(orbital, hamiltonian, occ)
                grad = gradient_function(orbital, hamiltonian, occ)

            direction = -self.__two_loop_recursion(grad)
            # Line search to find the optimal step size
            step_size, new_orbital, new_energy = self.__line_search( orbital, direction, grad, hamiltonian, occ)
            # orthonormalize 
            for i_s in range(new_orbital.shape[0]):
                for i_k in range(new_orbital.shape[1]):
                    new_orbital[i_s,i_k] = PH.redistribute(parallel_orthonormalize(PH.redistribute(new_orbital[i_s,i_k])), dim0=0, dim1=1)
            
            # recalculate energy and grad
            new_energy = energy_function(new_orbital, hamiltonian, occ)
            new_grad = gradient_function(new_orbital, hamiltonian, occ) 

            # set rho_prev and recalculate density using new_orbital
            self.rho_prev = self.density.get_density()

            new_orbital_ = np.zeros_like(new_orbital)
            # transpose
            for i_s in range(new_orbital.shape[0]):
                for i_k in range(new_orbital.shape[1]):
                    new_orbital_[i_s,i_k] = new_orbital[i_s,i_k].T
            
            self.density.calc_density(new_orbital_, occ, S = hamiltonian.get_S())

            ## Check convergence.
            self.converged = self.check_convg(
                occ, orbital, hamiltonian, print_energies
            )

            print(
                f"\n==================== [ END CYCLE {i_iter + 1} ] ===================="
            )
            if self.converged:
                orbital = new_orbital
                energy = new_energy
                
                den1 = self.density.get_density()

                hamiltonian.eigensolver.maxiter =[1000]
                eigval, eigvec  = hamiltonian.diagonalize()
                occ = self.occupation.get_occupation(eigval)

                energy_=0.0
                for i_s in range(hamiltonian.nspins):
                    for i_k, (kpt, weight) in enumerate(zip(*hamiltonian.kpoint.get_kpts_and_weights())):
                        energy_ +=sum(weight*occ[i_s,i_k]*eigval[i_s,i_k] )

                self.density.calc_density(eigvec, occ, S=hamiltonian.get_S())
                den2 = self.density.get_density()

                hamiltonian.calc_and_print_energies(self.density, eigval, eigvec, occ )
                
                if torch.abs(energy-energy_)>1e-3:
                    print( f"WARNING: energy from L-BFGS isn't same to the eigenvalue sum from eigensolver,{energy}, {energy_}")
            
                if torch.max((torch.abs(den1-den2) / den1)[den1>1e-4] ) > 1e-2: 
                    print( f"WARNING: density from L-BFGS isn't same to the density from eigensolver, {torch.max((torch.abs(den1-den2) / den1)[den1>1e-4] )}")

#                eye=torch.eye(eigvec[0,0].size(0), dtype=eigvec[0,0].dtype, device=eigvec[0,0].device) 
#                for i_s in range(orbital.shape[0]):
#                    for i_k in range(orbital.shape[1]):
#                        assert torch.mean(torch.abs(eigvec[i_s,i_k].conj()@orbital[i_s,i_k]-eye ) ) <1e-4, f"{i_s} {i_k}  orbital problem!\n{eigvec[i_s,i_k].conj()@orbital[i_s,i_k]}"

                # update final eigval, eigvec, and fermi_level
                self.eigvec = eigvec
                self.eigval = eigval
                self.fermi_level = self.occupation.fermi_level
                        
                break

            new_grad = gradient_function(new_orbital, hamiltonian, occ)

            s = new_orbital - orbital
            y = new_grad - grad
            self.__update(s, y)
    
            orbital = new_orbital
            energy = new_energy
            grad = new_grad
            print(energy)
        self.eigvec = orbital
        self.eigval = eigval
        self.energy = energy

        return orbital, energy

    def __line_search(self, orbital, direction, grad, hamiltonian, occ):
        alpha=self.param['alpha'] #1.0
        beta =self.param['beta']  #0.5
        sigma=self.param['sigma'] #1e-4
        energy = energy_function(orbital, hamiltonian, occ)
        grad_direction = dot(grad, direction)
        #grad_direction = torch.dot(grad.view(-1), direction.view(-1))
   
        new_orbital = np.zeros_like(orbital)
        kpt, weight = hamiltonian.kpoint.get_kpts_and_weights()

        occ_size = len(occ[0,0])
        for i in range(100):
            # orbital is numpy array (nspins, nibzkpts) 
            # each element poit torch.tensor (ngpts, nbands)
            # but grad and direction is torch.tensor( nspins*nibzkpts, gpts, nbands)
            for j, (i_s, i_k) in enumerate(itertools.product( range(orbital.shape[0]), range(orbital.shape[1]))):
                occupied = (occ[i_s,i_k]>1e-9)
                new_orbital[i_s,i_k] = orbital[i_s,i_k].clone()
                new_orbital[i_s,i_k][:,occupied] = orbital[i_s,i_k][:,occupied] + alpha * direction[:,j*occ_size:(j+1)*occ_size][:,occupied]
            new_energy = energy_function(new_orbital, hamiltonian, occ)
            assert False==torch.isnan(new_energy), "new_energy is nan!"
            sum_val =  sum(  [ weight[i_k]*occ[i_s,i_k]*grad_direction[j] for j, (i_s, i_k) in enumerate(itertools.product( range(orbital.shape[0]), range(orbital.shape[1]))) ] )
            if new_energy <= energy + sigma*alpha*sum_val.sum() :
                return alpha, new_orbital, new_energy
            alpha *= beta
        else:
            raise RuntimeError('line search does not meet convergence')
 
    def check_convg(self, occ, orbital, hamiltonian, print_energies):
        ## Relative density diff
        density_convg = False
        density_diff = hamiltonian.grid.integrate(
            (self.density.get_density() - self.rho_prev).sum(0).abs()
        )
        density_diff /= self.density.nelec
        self.density_diff = density_diff
        density_convg = abs(density_diff) <= self.density_tol
        print(f"Density diff (relative) = {density_diff} (/e)")

        ## Relative total energy diff
        energy_convg = False if print_energies else True
        total_energy = energy_function(orbital, hamiltonian, occ)
        self.total_energy_list.append(total_energy)
        energy_diff = self.total_energy_list[-1] - self.total_energy_list[-2]
        energy_diff /= self.density.nelec
        print(f"Total energy diff (relative) = {energy_diff} (Hartree/e)")
        energy_convg = abs(energy_diff) <= self.energy_tol

        if energy_convg and density_convg:
            return True
        else:
            return False

    
def energy_function(orbital, hamiltonian,occ):
    energy = 0.0
    for i_s in range(hamiltonian.nspins):
        for i_k, (kpt, weight) in enumerate(zip(*hamiltonian.kpoint.get_kpts_and_weights())):
            
            energy += weight*torch.sum(occ[i_s,i_k].view(1,-1)*orbital[i_s,i_k] * (hamiltonian[i_s,i_k] @ orbital[i_s,i_k]) )
    return energy

def gradient_function(orbital, hamiltonian,occ):
    grad = np.zeros((hamiltonian.nspins, hamiltonian.nkpts),dtype=object)
    for i_s in range(hamiltonian.nspins):
        for i_k, (kpt, weight) in enumerate(zip(*hamiltonian.kpoint.get_kpts_and_weights())):
            grad[i_s, i_k] = 2*weight*occ[i_s,i_k].view(1,-1)* (hamiltonian[i_s,i_k] @ orbital[i_s,i_k]) 
            #grad[i_s, i_k] = orbital[i_s,i_k]@(grad[i_s, i_k].conj().T@orbital[i_s,i_k]) # orbital projection
    grad = torch.cat ([ g for g in grad.reshape(-1) ] , dim=-1)
    return grad

