import torch
from pathlib import Path 

from gospel.ParallelHelper import ParallelHelper as PH
from gospel.Mixing import Mixing
from gospel.util import Timer


class SCF:
    """Self-consistent field (SCF) method

    :type  convg: dict
    :param convg:
        dictionary of convergence conditions:
        'scf_maxiter', 'energy_tol', 'density_tol', 'orbital_energy_tol'
    :type  occupation: gospel.Occupation
    :param occupation:
        Occupation class object
    :type  mixing_params: dict
    :param mixing_params:
        dictionary of parameters for mixing
    :type  density: gospel.Density
    :param density:
        Density class object
    """

    def __init__(self, convg, occupation, mixing_params, density):
        ## Convergence criteria
        self.scf_maxiter = convg.get("scf_maxiter")
        self.energy_tol = convg.get("energy_tol")
        self.density_tol = convg.get("density_tol")
        self.orbital_energy_tol = convg.get("orbital_energy_tol")
        self.bands = convg.get("bands")

        if isinstance(self.bands, int):
            self.nocc = self.bands
        elif self.bands in ["occupied", "all"]:
            # if bands is 'occupied', self.nocc will be updated during SCF.
            self.nocc = None
        else:
            raise RuntimeError(
                f"convg['bands'] should be one of 'occupied' or 'all' or int (current: {self.bands})"
            )

        self.occupation = occupation
        self.mixing = Mixing(**mixing_params)
        print(self.mixing)
        self.density = density

        self.history_dir = None

        self.eigvec = None
        self.eigval = None
        self.energy = None
        self.fermi_level = None

        self.total_energy_list = [torch.nan]
        self.eigval_sum = [torch.nan]
        self.converged = False
        self.rho_prev = None
        self.density_diff = None

        self.diag_tol = None
        return

    def __str__(self):
        s = str()
        s += "\n============================== [ SCF ] =============================="
        s += f"\n* scf_maxiter        : {self.scf_maxiter}"
        s += f"\n* energy_tol         : {self.energy_tol}"
        s += f"\n* density_tol        : {self.density_tol}"
        s += f"\n* orbital_energy_tol : {self.orbital_energy_tol}"
        s += f"\n* bands              : {self.bands}"
        s += "\n=====================================================================\n"
        return s

    def iterate(self, hamiltonian, print_energies=False):
        density = self.density
        mixing = self.mixing

        for i_iter in range(self.scf_maxiter):
            Timer.start("SCF iter.", True)
            print(
                f"\n==================== [ SCF CYCLE {i_iter + 1} ] ===================="
            )
            # Set initial guess density and update Hamiltonian
            if i_iter == 0:
                init_density = density.init_density()
                density.set_density(init_density)
                hamiltonian.update(density)
                if mixing.what == "potential":
                    mixing.append(hamiltonian.V_loc)
                else:  # mixing.what == "density"
                    mixing.append(init_density)

            # Adjust diagonalization convergence tolerance
            # For first iteration, set diag_tol to 0.1 to avoid over-diagonalization
            # For subsequent iterations, set diag_tol proportional to density_diff
            if i_iter == 0:
                self.diag_tol = 0.1
            else:
                self.diag_tol = min(0.1 * self.density_diff, self.diag_tol)
            print(f"Diagonalization convg_tol is set to {self.diag_tol}.")

            # Save the previous density to check convergence.
            self.rho_prev = density.get_density()

            # Diagonalize Hamiltonian and get orbitals.
            # Note: For first iteration with bands='occupied', self.nocc is None,
            # so davidson will check all bands initially
            ret_hist = self.history_dir is not None
            diag_out = hamiltonian.diagonalize(
                convg_tol=self.diag_tol, i_scf=i_iter, bands=self.nocc, retHistory=ret_hist
            )
            
            if ret_hist:
                eigval, eigvec, histories = diag_out
                if PH.is_master():
                    out_dir = Path(self.history_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for i_s in range(histories.shape[0]):
                        for i_k in range(histories.shape[1]):
                            out_path = out_dir / f"scf_{i_iter + 1:03d}_s{i_s}_k{i_k}.pt"
                            torch.save(histories[i_s, i_k], out_path)
            else:
                eigval, eigvec = diag_out
            # eigval, eigvec = hamiltonian.diagonalize(
            #     convg_tol=self.diag_tol, i_scf=i_iter, bands=self.nocc
            # )

            # Fill the occupation.
            occ = self.occupation.get_occupation(eigval)
            # Update nocc for 'occupied' mode
            if self.bands == "occupied":
                # Consider all spins and k-points for accurate occupied count
                occ_threshold = 1e-8
                # occ shape: (nspins, nibzkpts, nbands)
                # Find maximum occupied states across all spins and k-points
                nocc_per_kpt = (occ > occ_threshold).sum(dim=-1)  # sum over bands
                self.nocc = nocc_per_kpt.max().item()  # max over spins and k-points
                self.nocc = max(self.nocc, 1)  # Ensure at least one band is considered
                print(f"Number of bands to check convergence: {self.nocc}")

            # Calculate new charge density from orbitals.
            density.calc_density(eigvec, occ, S=hamiltonian.get_S())

            # Check convergence.
            self.converged = self.check_convg(
                occ, eigval, eigvec, density, hamiltonian, print_energies
            )

            # Save density or potential to mixing.stack and Update hamiltonian.
            if not self.converged:
                if mixing.what == "density":
                    mixing.append(density.get_density())
                    density.set_density(mixing.get_mixed_one())
                    hamiltonian.update(density)
                elif mixing.what == "potential":
                    hamiltonian.update(density)
                    mixing.append(hamiltonian.V_loc)
                    hamiltonian.V_loc = mixing.get_mixed_one()
                else:
                    raise RuntimeError(
                        f"mixing.what should be one of 'density' or 'potential' (current: {mixing.what})"
                    )

            # Print elapsed time for each scf step.
            Timer.stop("SCF iter.", True, True)
            print(
                f"\n==================== [ END CYCLE {i_iter + 1} ] ===================="
            )

            if self.converged:
                break

        if self.converged:
            print(f"SCF CONVERGED with {i_iter + 1} iters!!")
            self.energy = hamiltonian.calc_and_print_energies(
                density, eigval, eigvec, occ
            )
        else:
            print(f"SCF NOT CONVERGED!! (SCF MAX CYCLE={self.scf_maxiter})")
            print("You can change the convergence options.")

        ## Save final density, eigenvectors, and eigenvalues
        self.eigvec = eigvec
        self.eigval = eigval
        self.fermi_level = self.occupation.fermi_level
        return

    def check_convg(self, occ, eigval, eigvec, density, hamiltonian, print_energies):
        """
        Print relative differences of orbital energies, total energies, and densities
        with previous step's, and check their convergence.

        The three tolerances for orbital energy, total energy, and density are all
        proportional to the number of electrons.
        """
        ## Relative orbital energy sum diff
        orbital_energy_convg = False
        new_eigval_sum = (occ * eigval).sum()
        self.eigval_sum.append(new_eigval_sum)
        eigval_sum_diff = self.eigval_sum[-1] - self.eigval_sum[-2]
        eigval_sum_diff /= density.nelec
        print(f"Orbital energy sum diff (relative) = {eigval_sum_diff} (Hartree/e)")
        orbital_energy_convg = abs(eigval_sum_diff) <= self.orbital_energy_tol

        ## Relative density diff
        density_convg = False
        density_diff = hamiltonian.grid.integrate(
            (density.get_density() - self.rho_prev).sum(0).abs()
        )
        density_diff /= density.nelec
        self.density_diff = density_diff
        density_convg = abs(density_diff) <= self.density_tol
        print(f"Density diff (relative) = {density_diff} (/e)")

        ## Relative total energy diff
        energy_convg = False if print_energies else True
        if print_energies:
            total_energy = hamiltonian.calc_and_print_energies(
                density, eigval, eigvec, occ
            )
            self.total_energy_list.append(total_energy)
            energy_diff = self.total_energy_list[-1] - self.total_energy_list[-2]
            energy_diff /= density.nelec
            print(f"Total energy diff (relative) = {energy_diff} (Hartree/e)")
            energy_convg = abs(energy_diff) <= self.energy_tol

        if energy_convg and density_convg and orbital_energy_convg:
            return True
        else:
            return False
