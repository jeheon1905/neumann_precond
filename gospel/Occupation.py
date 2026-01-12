import torch
from gospel.smearing_functions import *


class Occupation:
    def __init__(
        self,
        nelec,
        nspins,
        nbands,
        nkpts,
        scaled_kpts,
        weights,
        smearing=None,
        fix_magmom=True,
        magmom=0.0,
        temperature=1160.4518,
        MP_order=2,
    ):
        """Calculate occupation numbers using various smearing methods.

        Arguments)
        nelec         : float   | # of electrons
        nspins        : int     | # of spins (1 or 2)
        nbands        : int     | # of orbitals
        nkpts         : int     | total # of k-points
        scaled_kpts   : ndarray | k-points
        weights       : ndarray | weights of each k-point
        smearing      : str     | smearing method  e.g., None, 'Fermi-Dirac', 'Gaussian', 'Methfessel-Paxton', 'Marzari-Vanderbilt'
        fix_magmom    : bool    | whether fix magmom or not
        magmom        : float   | magnetic momentum
        temperature   : float   | temperature (Kelvin)
        MP_order      : int     | order of Methfessel-Paxton method

        Notice that occupation values are unscaled for the number of k-points. So, the sum of occupation values equals to nkpts * nelec.
        """
        self.__nelec = nelec
        self.__nspins = nspins
        self.__nbands = nbands

        self.__nkpts = nkpts
        self.__scaled_kpts = scaled_kpts
        self.__weights = torch.from_numpy(weights)   # -> torch.Tensor 로 바꾸기!!
        self.__nibzkpts = len(self.__scaled_kpts)

        self.__smearing = smearing
        self.__fix_magmom = fix_magmom
        self.__magmom = magmom
        if self.__nspins == 1:
            assert (
                self.__magmom == 0.0
            ), f"magmom can't be {self.__magmom} in 'unpolarized' calculation."
        self.__temperature = temperature
        self.__MP_order = MP_order  # Only used for 'Methfessel-Paxton' smearing

        self.fermi_level = None  # "calculated only when smearing is used"
        return

    def __str__(self):
        s = str()
        s += "\n========================== [ Occupation ] =========================="
        s += f"\n* smearing    : {self.__smearing}"
        if self.__nspins == 2:
            s += f"\n* magmom      : {self.__magmom}"
            s += f"\n* fix_magmom  : {self.__fix_magmom}"
        if self.__smearing == "Methfessel-Paxton":
            s += f"\n* MP_order    : {self.__MP_order}"
        if self.__smearing:
            s += f"\n* temperature : {self.__temperature}"
        s += f"\n* nelec       : {self.__nelec}"
        s += f"\n* nspins      : {self.__nspins}"
        s += f"\n* nbands      : {self.__nbands}"
        s += f"\n* nkpts       : {self.__nkpts}"
        s += "\n====================================================================\n"
        return s

    def get_occupation(self, eigval):
        """Get occupation number
        :type  eigval: torch.Tensor
        :param eigval:
            eigenvalues, shape=(nspins, nibzkpts, nbands)

        :rtype: torch.Tensor
        :return:
            occupation number, shape=(nspins, nibzkpts, nbands)
        """
        if self.__smearing is None or self.__temperature == 0.0:
            occ = self.calc_no_smearing_occ(eigval)
        elif self.__smearing == "Fermi-Dirac":
            occ = self.calc_smearing_occ(eigval, fermi_dirac_smearing_function)
        elif self.__smearing == "Gaussian":
            occ = self.calc_smearing_occ(eigval, gaussian_smearing_function)
        elif self.__smearing == "Methfessel-Paxton":
            occ = self.calc_smearing_occ(eigval, mp_smearing_function)
        elif self.__smearing == "Marzari-Vanderbilt":
            occ = self.calc_smearing_occ(eigval, cold_smearing_function)
        else:
            assert 0, str(self.__smearing) + " is not available."

        ## Print eigenvalues and occupation
        # ASE band gap은 index 0부터 시작하므로, 여기서도 0부터 시작하도록 하기?.
        print("\n[ PRINT EIGENVALUES ]")
        width = 10
        for i_s in range(self.__nspins):
            print("( spin alpha )") if i_s == 0 else print("( spin beta )")
            kpts = self.__scaled_kpts
            for i_k in range(self.__nibzkpts):
                print(f"* {i_k+1} th kpt (={kpts[i_k]})")
                print(f"{'num':<{width}} {'occ':<{width}} value")
                for i in range(self.__nbands):
                    print(
                        f"{i+1:<{width}} {torch.round(occ[i_s,i_k,i] * self.__nkpts, decimals=7):<{width}} {eigval[i_s,i_k,i]}"
                    )
                print("\n", flush=True)
        return occ

    def calc_smearing_occ(self, eigval, smearing_function, tol=1e-7):
        if self.__nspins == 2 and self.__fix_magmom == True:
            occupation = torch.zeros_like(eigval)
            for i_s in range(self.__nspins):
                occupation[i_s] = self._calc_smearing_occ(
                    eigval[i_s], smearing_function, tol, i_s
                )
        else:
            occupation = self._calc_smearing_occ(eigval, smearing_function, tol)
        return occupation

    def _calc_smearing_occ(self, eigval, smearing_function, tol=1e-7, i_s=None):
        occupation = torch.zeros_like(eigval)
        max_iter = 200
        fermi_converged = False

        if i_s == 0:
            nelec = (self.__nelec + self.__magmom) / self.__nspins
        elif i_s == 1:
            nelec = (self.__nelec - self.__magmom) / self.__nspins
        elif i_s is None:
            nelec = self.__nelec
        else:
            assert False, f"i_s should be one of 0, 1, and None."
        # max_val = eigval.max()
        max_val = eigval.max() + 0.1  # added some small value to handle the case with no virtual orbitals
        min_val = eigval.min()
        smearing_iter = 0
        while smearing_iter < max_iter:
            fermi_level = (min_val + max_val) / 2
            energy_diff = eigval - fermi_level
            occupation = smearing_function(
                energy_diff, self.__temperature, self.__MP_order
            )
            occupation *= (
                2 / self.__nspins * self.__weights.reshape(len(self.__weights), -1).to(occupation.device)
            )

            check_nelec = occupation.sum()
            if abs(check_nelec - nelec) < tol:
                fermi_converged = True
                self.fermi_level = fermi_level
                break
            elif check_nelec < nelec:
                min_val = fermi_level
            else:
                max_val = fermi_level
            smearing_iter += 1

        if not fermi_converged:
            print(f"fermi-level didn't converge. (max_iter={max_iter})")

        return occupation

    def calc_no_smearing_occ(self, eigval):
        self.__temperature = 0.0
        print(f"Occupation: smearing is not used. 'temperature' is set to 0.0")
        return self.calc_smearing_occ(eigval, fermi_dirac_smearing_function)

    @property
    def smearing(self):
        return self.__smearing

    @property
    def nelec(self):
        return self.__nelec

    @property
    def nspins(self):
        return self.__nspins

    @property
    def nbands(self):
        return self.__nbands

    @property
    def nkpts(self):
        return self.__nkpts

    @property
    def nibzkpts(self):
        return self.__nibzkpts

    @property
    def fix_magmom(self):
        return self.__fix_magmom

    @property
    def magmom(self):
        return self.__magmom

    @property
    def temperature(self):
        return self.__temperature

if __name__ == "__main__":
    from gospel.Kpoint import Kpoint
    from ase.dft.kpoints import monkhorst_pack

    if True:
        print("=========== Test Occupation class (unpolarized) ==========\n")
        kpts_mesh = torch.tensor([1, 1, 1])
        kpts = monkhorst_pack(kpts_mesh)
        weights = torch.full((len(kpts),), 1 / len(kpts))
        nkpts = kpts_mesh.prod()
        smearing = None
        # smearing   = 'Fermi-Dirac'
        temperature = 0.0
        # temperature  = 1.0
        # nelec = 6.0
        nelec = 6.1
        nbands = 6
        spin = "unpolarized"
        nspins = 1 if spin == "unpolarized" else 2
        print("=" * 15 + " Input " + "=" * 15)
        print("* kpts_mesh    = ", kpts_mesh)
        print("* kpts         = \n", kpts)
        print("* weights      = ", weights)
        print("* nelec        = ", nelec)
        print("* nbands       = ", nbands)
        print("* spin         = ", spin)
        print("* nspins       = ", nspins)
        print("* smearing     = ", smearing)
        print("* temperature  = ", temperature)
        print("=" * 15 + " Input " + "=" * 15)

        occupation = Occupation(
            smearing=smearing,
            temperature=temperature,
            nelec=nelec,
            nbands=nbands,
            nkpts=kpts_mesh.prod(),
            scaled_kpts=kpts,
            weights=weights,
            nspins=nspins,
        )

        eigval = torch.zeros(nspins, nkpts, nbands, dtype=torch.float64)
        for i_s in range(nspins):
            for k in range(nkpts):
                eigval[i_s, k] = (torch.arange(0.0, 1.0, 0.1) + 0.01 * k)[:nbands]
        print("* eigval = ", eigval)

        occ = occupation.get_occupation(eigval)
        print("\nocc = ", occ)
        print("========================================================\n")

    if True:
        print("=========== Test Occupation class (polarized) ==========\n")
        kpts_mesh = torch.tensor([1, 1, 1])
        kpts = monkhorst_pack(kpts_mesh)
        weights = torch.full((len(kpts),), 1 / len(kpts))
        nkpts = kpts_mesh.prod()
        smearing = None
        # smearing        = 'Fermi-Dirac'
        temperature = 0.0
        # temperature     = 1.0
        nelec = 6.1
        # nelec   = 6.0
        nbands = 6
        spin = "polarized"
        nspins = 1 if spin == "unpolarized" else 2
        print("=" * 15 + " Input " + "=" * 15)
        print("* kpts_mesh   = ", kpts_mesh)
        print("* kpts        = \n", kpts)
        print("* weights     = ", weights)
        print("* nelec       = ", nelec)
        print("* nbands      = ", nbands)
        print("* spin        = ", spin)
        print("* nspins      = ", nspins)
        print("* smearing    = ", smearing)
        print("* temperature = ", temperature)
        print("=" * 15 + " Input " + "=" * 15)

        occupation = Occupation(
            smearing=smearing,
            temperature=temperature,
            nelec=nelec,
            nbands=nbands,
            nkpts=kpts_mesh.prod(),
            scaled_kpts=kpts,
            weights=weights,
            nspins=nspins,
        )

        eigval = torch.zeros(nspins, nkpts, nbands, dtype=torch.float64)
        for i_s in range(nspins):
            for k in range(nkpts):
                eigval[i_s, k] = (torch.arange(0.0, 1.0, 0.1) + 0.01 * k)[:nbands]
        print("* eigval = ", eigval)

        occ = occupation.get_occupation(eigval)
        print("\nocc = ", occ)
        print("========================================================\n")

    if True:
        print("=========== test occupation class (polarized) ==========\n")
        kpts_mesh = torch.tensor([3, 1, 1])
        kpts = torch.tensor([[0, 0, 0], [1 / 3, 0, 0]])
        weights = torch.tensor([1 / 3, 2 / 3])
        nibzkpts = len(kpts)
        smearing = None
        # smearing        = 'Fermi-Dirac'
        temperature = 0.0
        # temperature     = 1.0
        # nelec   = 6.1
        nelec = 6.0
        nbands = 6
        spin = "polarized"
        nspins = 1 if spin == "unpolarized" else 2
        print("=" * 15 + " input " + "=" * 15)
        print("* kpts_mesh   = ", kpts_mesh)
        print("* kpts        = \n", kpts)
        print("* weights     = ", weights)
        print("* nelec       = ", nelec)
        print("* nbands      = ", nbands)
        print("* spin        = ", spin)
        print("* nspins      = ", nspins)
        print("* smearing    = ", smearing)
        print("* temperature = ", temperature)
        print("=" * 15 + " input " + "=" * 15)

        occupation = Occupation(
            smearing=smearing,
            temperature=temperature,
            nelec=nelec,
            nbands=nbands,
            nkpts=kpts_mesh.prod(),
            scaled_kpts=kpts,
            weights=weights,
            nspins=nspins,
        )

        eigval = torch.zeros(nspins, nibzkpts, nbands, dtype=torch.float64)
        for i_s in range(nspins):
            for k in range(nibzkpts):
                eigval[i_s, k] = (torch.arange(0.0, 1.0, 0.1) + 0.01 * k)[:nbands]
        print("* eigval = \n", eigval)

        occ = occupation.get_occupation(eigval)

        print("\nocc = ", occ)
        print("========================================================\n")
