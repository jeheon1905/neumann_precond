from typing import List
import re
import xml.etree.ElementTree as elemTree
from math import ceil
import numpy as np
import torch
from scipy.interpolate import interp1d


def get_zero_convg_idx(arr, tol=1e-7):
    """Obtain an index whose value converges less than tolerance value."""
    is_zero = np.abs(arr) < tol
    idx = 0
    for i in range(len(arr) - 1):
        if is_zero[-1 - i] == True:
            idx += 1
        if is_zero[-1 - i] != is_zero[-2 - i]:
            break
    if idx == 0:
        return -1  # if not converge to zero -> return last index
    return len(arr) - idx


def element_to_ndarray(element):
    """Make ndarray from the element of ElementTree."""
    func = lambda x: float(x) if x != "" else None
    return np.asarray(list(map(func, re.split("\s+|\n", element.text.strip()))))


class Read_UPF:
    """Read and preprocess an UPF file.

    :type  filename: str
    :param filename:
        UPF filename
    """

    def __init__(self, filename):
        assert isinstance(filename, str)
        self.filename = filename

        ## preprocessing : Replace '&' with white space. If '&' is contained, 'not well-formed' ParseError occurs.
        content = open(filename).read().replace("&", " ")
        tree = elemTree.fromstring(content)

        ## Read PP_HEADER (elements, zval, num_proj, pseudo_type)
        PP_HEADER = tree.find("PP_HEADER")
        element = PP_HEADER.attrib["element"].strip()
        z_valence = float(PP_HEADER.attrib["z_valence"])
        number_of_proj = int(PP_HEADER.attrib["number_of_proj"])
        pseudo_type = PP_HEADER.attrib["pseudo_type"].upper()

        ## Read PP_MESH (PP_R, PP_RAB)
        PP_MESH = tree.find("PP_MESH")
        PP_R = element_to_ndarray(PP_MESH.find("PP_R"))
        PP_RAB = element_to_ndarray(PP_MESH.find("PP_RAB"))
        if "size" in PP_MESH.find("PP_R"):
            assert int(PP_MESH.find("PP_R").attrib["size"]) == len(
                PP_R
            ), "reading PP_R caused the problem. please check pseudopotential files"
        if "size" in PP_MESH.find("PP_RAB"):
            assert int(PP_MESH.find("PP_RAB").attrib["size"]) == len(
                PP_R
            ), "reading PP_RAB caused the problem. please check pseudopotential files"

        ## Read PP_NONLOCAL (PP_BETA, PP_DIJ)
        PP_NONLOCAL = tree.find("PP_NONLOCAL")
        angmom = []
        beta_cutoff = []
        for nl in PP_NONLOCAL:
            if nl.tag.startswith("PP_BETA"):
                if "." in nl.tag:
                    name, num = nl.tag.split(".")
                    assert name == "PP_BETA"
                    attr = nl.attrib
                    angmom.append(int(attr["angular_momentum"]))
                    # beta_cutoff.append(float(attr["cutoff_radius"]))  # some errors occur in TM pseudopotential
                    # beta_cutoff.append(PP_R[int(attr["cutoff_radius_index"])])
                    _cutoff = PP_R[int(attr["cutoff_radius_index"])]
                    _cutoff = ceil(_cutoff * 1e+2) / 1e+2
                    beta_cutoff.append(_cutoff)
                else:
                    assert False, "Error in reading PP_BETA"
            else:
                assert nl.tag == "PP_DIJ"

        ## Read PP_BETA
        PP_BETA = []
        for i in range(number_of_proj):
            PP_BETA.append(element_to_ndarray(PP_NONLOCAL.find(f"PP_BETA.{i+1}")))
        if number_of_proj:
            PP_BETA = np.vstack(PP_BETA)

        ## Read PP_DIJ and local_cutoff
        PP_DIJ = element_to_ndarray(PP_NONLOCAL.find("PP_DIJ"))
        if number_of_proj != 0:
            PP_DIJ *= 2  # 1 Ry^{-1} = 2 Bohr^{-1}

        ## Read PP_LOCAL
        PP_LOCAL = element_to_ndarray(tree.find("PP_LOCAL"))
        local_cutoff = beta_cutoff[0] if len(beta_cutoff) > 0 else 5.0  # large enough but not too large

        ## Read PP_RHOATOM
        PP_RHOATOM = element_to_ndarray(tree.find("PP_RHOATOM"))

        ## Read PP_NLCC and nlcc_cutoff
        if tree.find("PP_NLCC") is None:
            PP_NLCC = None
        else:
            PP_NLCC = element_to_ndarray(tree.find("PP_NLCC"))

        ## Read PP_AUG (only used for LUSP method)
        if tree.find("PP_AUG") is None:
            PP_AUG = None
        else:
            assert pseudo_type == "LUSP"
            PP_AUG = element_to_ndarray(tree.find("PP_AUG"))

        ## Divide with R and convert to atomic unit (a.u.)
        PP_LOCAL *= 0.5  # Rydberg to Hartree (1 Ry = 0.5 Hartree)
        if number_of_proj:
            PP_BETA = np.divide(
                PP_BETA, PP_R, out=np.zeros_like(PP_BETA), where=PP_R != 0
            )
            PP_BETA *= 0.5  # unit conversion from Ryd*Bohr^-0.5 to Hartree*Bohr^-0.5
        PP_RHOATOM = np.divide(
            PP_RHOATOM,
            4 * np.pi * PP_R**2,
            out=np.zeros_like(PP_RHOATOM),
            where=PP_R != 0,
        )
        PP_RHOATOM[PP_R < 0.01] = PP_RHOATOM[PP_R > 0.01][0]  # treat small r region of PP_RHOATOM

        ## Define radial grid
        max_radius = 10.0
        # Rmin, Rmax, dR = 0.0, min(PP_R[-1], max_radius), 0.01
        Rmin, Rmax, dR = 0.0, max_radius, 0.01  # Fix: set Rmax as 10.

        R = np.arange(Rmin, Rmax + 1e-7, dR)

        ## Treat PP_R=0 point.
        ## values at r=0 are removed and recalculated by interpolation
        if PP_R[0] == 0.0:
            PP_R = PP_R[1:]
            PP_LOCAL = PP_LOCAL[1:]
            PP_RHOATOM = PP_RHOATOM[1:]
            if PP_NLCC is not None:
                PP_NLCC = PP_NLCC[1:]
            if number_of_proj:
                PP_BETA = PP_BETA[:, 1:]
            if PP_AUG is not None:
                PP_AUG = PP_AUG[1:]

        ## Interpolation to our defined radial grid.
        interp_option = {
            "kind": "cubic",
            "bounds_error": False,
            "fill_value": "extrapolate",
        }
        local_interp = interp1d(PP_R, PP_LOCAL, **interp_option)
        PP_LOCAL = local_interp(R)
        del local_interp
        if number_of_proj:
            beta_interp = interp1d(PP_R, PP_BETA, **interp_option)
            PP_BETA = beta_interp(R)
            del beta_interp
        rhoatom_interp = interp1d(PP_R, PP_RHOATOM, **interp_option)
        PP_RHOATOM = rhoatom_interp(R)
        del rhoatom_interp
        if PP_NLCC is not None:
            nlcc_interp = interp1d(PP_R, PP_NLCC, **interp_option)
            PP_NLCC = nlcc_interp(R)
            del nlcc_interp
        if PP_AUG is not None:
            aug_interp = interp1d(PP_R, PP_AUG, **interp_option)
            PP_AUG = aug_interp(R)
            del aug_interp

        ## extrapolate with cutoff radii: local and beta
        PP_LOCAL[R > local_cutoff] = - z_valence / R[R > local_cutoff]
        for i in range(number_of_proj):
            PP_BETA[i][R > beta_cutoff[i]] = 0.0

        ## Calculate cutoff radii of PP_RHOATOM, PP_NLCC
        """rho_cutoff=0.0 일 때는, get_zero_convg_idx 로 구해서 쓰게 바꾸기."""
        if "rho_cutoff" in PP_HEADER.attrib:
            rho_cutoff = float(PP_HEADER.attrib["rho_cutoff"])
        else:
            rho_cutoff = None
        if rho_cutoff == 0.0 or rho_cutoff is None:
            rho_cutoff = R[get_zero_convg_idx(PP_RHOATOM)]
        if rho_cutoff <= 0.0 or rho_cutoff > max_radius:
            print(f"Warning: rho_cutoff is {rho_cutoff}.", end=" ")
            rho_cutoff = min(R[-1], max_radius)
            print(f"'rho_cutoff' is set to {rho_cutoff}")
        # if "rho_cutoff" in PP_HEADER.attrib:
        #     rho_cutoff = float(PP_HEADER.attrib["rho_cutoff"])
        # else:
        #     rho_cutoff = R[get_zero_convg_idx(PP_RHOATOM)]
        # if rho_cutoff >= R[-1] or rho_cutoff == 0.0:
        #     print(
        #         f"Warning: 'rho_cutoff'(={rho_cutoff}) value is too large or small. 'rho_cutoff' is set to {R[-1]}."
        #     )
        #     rho_cutoff = R[-1]
        if PP_NLCC is not None:
            nlcc_cutoff = R[get_zero_convg_idx(PP_NLCC)]
            assert nlcc_cutoff < R[-1], f"nlcc_cutoff is too long (={nlcc_cutoff})"
        else:
            nlcc_cutoff = None
        if PP_AUG is not None:
            aug_cutoff = R[get_zero_convg_idx(PP_AUG)]
            assert aug_cutoff < R[-1], f"aug_cutoff is too long (={aug_cutoff})"
        else:
            aug_cutoff = None

        ## Check cutoff radii is valid
        assert (
            np.max(list(filter(None, [local_cutoff, rho_cutoff, nlcc_cutoff]))) <= R[-1]
        ), f"local_cutoff, rho_cutoff, nlcc_cutoff: {local_cutoff}, {rho_cutoff}, {nlcc_cutoff}"
        if beta_cutoff:
            assert np.max(beta_cutoff) <= R[-1]

        ## Set instance variables
        self.__element = element
        self.__zval = z_valence
        self.__num_proj = number_of_proj
        self.__pseudo_type = pseudo_type
        self.__angmom = angmom
        self.__local_cutoff = local_cutoff
        self.__beta_cutoff = beta_cutoff
        self.__rho_cutoff = rho_cutoff
        self.__nlcc_cutoff = nlcc_cutoff
        self.__aug_cutoff = aug_cutoff

        self.__Rmin = Rmin
        self.__Rmax = Rmax
        self.__dR = dR
        self.__dR = np.ones_like(R) * dR  ## 임시!!.
        self.__R = R
        self.__local = PP_LOCAL
        self.__beta = PP_BETA
        self.__dij = PP_DIJ
        self.__rhoatom = PP_RHOATOM
        self.__nlcc = PP_NLCC
        self.__aug = PP_AUG
        return

    @property
    def element(self):
        return self.__element

    @property
    def zval(self):
        return self.__zval

    @property
    def num_proj(self):
        return self.__num_proj

    @property
    def pseudo_type(self):
        return self.__pseudo_type

    @property
    def angmom(self):
        return self.__angmom

    @property
    def local_cutoff(self):
        return self.__local_cutoff

    @local_cutoff.setter
    def local_cutoff(self, inp):
        self.__local_cutoff = inp
        return

    @property
    def beta_cutoff(self):
        return self.__beta_cutoff

    @beta_cutoff.setter
    def beta_cutoff(self, inp):
        self.__beta_cutoff = inp
        return

    @property
    def rho_cutoff(self):
        return self.__rho_cutoff

    @property
    def nlcc_cutoff(self):
        return self.__nlcc_cutoff

    @property
    def aug_cutoff(self):
        return self.__aug_cutoff

    @property
    def R(self):
        return self.__R

    @property
    def RAB(self):
        # def dR(self):
        # return self.__RAB
        return self.__dR

    @property
    def local(self):
        return self.__local

    @local.setter
    def local(self, inp):
        self.__local = inp
        return

    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, inp):
        self.__beta = inp
        return

    @property
    def dij(self):
        return self.__dij

    @property
    def rhoatom(self):
        return self.__rhoatom

    @property
    def nlcc(self):
        return self.__nlcc

    @property
    def aug(self):
        return self.__aug


class UPF:
    """Process Read_UPF objects.

    :type  filenames: list
    :param filenames:
        list of UPF filenames

    **Example**

    >>> filenames = ["H.upf", "C.upf"]
    >>> upf = UPF(filenames)
    >>> local_potential_of_H = upf["H"].local
    >>> local_potential_of_C = upf["C"].local
    """
    def __init__(
        self,
        filenames: List[str],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        assert isinstance(filenames, list)
        self.__filenames = filenames
        # TODO: Implement device management
        # TODO: Change all variables to torch tensors
        self.device = device

        upfs = [Read_UPF(filename) for filename in filenames]

        self.__elements = [upf.element for upf in upfs]
        assert len(self.__elements) == len(
            set(self.__elements)
        ), "Multiple UPF files for the same elements are detected"
        pseudo_type = np.array([upf.pseudo_type for upf in upfs])
        assert np.all(
            pseudo_type[0] == pseudo_type
        ), "'pseudo_type' is different between elements."
        self.__pseudo_type = pseudo_type[0]
        self.__upfs = {upf.element: upf for upf in upfs}
        return

    def __getitem__(self, key):
        if key not in self.__elements:
            raise KeyError
        else:
            return self.__upfs[key]

    @property
    def filenames(self):
        return self.__filenames

    @property
    def elements(self):
        return self.__elements

    @property
    def pseudo_type(self):
        return self.__pseudo_type

    @property
    def num_projs(self):
        return {e: self.__upfs[e].num_proj for e in self.__elements}

    @property
    def local(self):
        return {e: self.__upfs[e].local for e in self.__elements}

    @local.setter
    def local(self, inp):
        for el in self.__elements:
            assert len(inp[el]) == len(self.__upfs[el].local)
            self.__upfs[el].local = inp[el]
        return

    @property
    def local_cutoff(self):
        return {e: self.__upfs[e].local_cutoff for e in self.__elements}

    @local_cutoff.setter
    def local_cutoff(self, inp):
        for el in self.__elements:
            self.__upfs[el].local_cutoff = inp[el]
        return

    @property
    def beta(self):
        return {e: self.__upfs[e].beta for e in self.__elements}

    @beta.setter
    def beta(self, inp):
        for el in self.__elements:
            self.__upfs[el].beta = inp[el]
        return

    @property
    def beta_cutoff(self):
        return {e: self.__upfs[e].beta_cutoff for e in self.__elements}

    @beta_cutoff.setter
    def beta_cutoff(self, inp):
        for el in self.__elements:
            self.__upfs[el].beta_cutoff = inp[el]
        return


if __name__ == "__main__":
    filenames = [
        "/appl/ACE-Molecule/DATA/PSEUDOPOTENTIALS_NC/Si.pbe-n-nc.UPF",  # logarithmic grid
        # "/appl/ACE-Molecule/DATA/sg15_oncv_upf_2020-02-06/Si_ONCV_PBE-1.2.upf",
        # "/appl/ACE-Molecule/DATA/ONCV/nc-sr-04_pbe_standard/Si.upf",
    ]
    # filenames = ["/home/jhwoo/PP/pz.0.3.1/PSEUDOPOTENTIALS_NC/H.pz-n-nc.UPF"]  # logarithmic grid
    upfs = UPF(filenames)
    el = upfs.elements[0]
    upf = upfs[el]
    R = upf.R
    local = upf.local
    betas = upf.beta

    from gospel.FdOperators import calc_deriv_1D
    import matplotlib.pyplot as plt

    plt.plot(R, local, label="Vloc")
    plt.plot(R, calc_deriv_1D(R, local), label="d Vloc")
    for i, beta in enumerate(betas):
        plt.plot(R, beta, label=f"beta.{i}")
        plt.plot(R, calc_deriv_1D(R, beta), label=f"d beta.{i}")

    plt.axvline(R[0], color="k")
    plt.axvline(R[1], color="k")
    plt.axvline(R[-1], color="k")
    plt.axvline(R[-2], color="k")
    plt.axhline(0, color="k")
    plt.legend()
    plt.show()
