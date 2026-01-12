import numpy as np
import spglib


class Kpoint:
    """Sampling K-point

    :type  atoms: ase.Atoms
    :param atoms:
        Atoms class object
    :type  kpts: tuple or np.ndarray
    :param kpts:
        k-points mesh and gamma-centered,
        e.g., kpts=(4, 4, 4, "gamma")  # gamma-centered Monkhorst-Pack sampling
        e.g., kpts=[(kx1, ky1, kz1), (kx2, ky2, kz2), ...]  # list of k-points
    :type  kpts_and_weights: list or np.ndarray
    :param kpts_and_weights:
        directly entered kpoints and weights,
        e.g., kpts_and_weights=[ [[kx1, ky1, kz1], [kx2, ky2, kz2], ...], [w1, w2, ...] ]
    :type  group_sym: bool
    :param group_sym:
        whether making IBZ using group symmetry
    :type  inv_sym: bool
    :param inv_sym:
        whether makking IBZ using inversion symmetry
    """

    def __init__(
        self, atoms, kpts=None, kpts_and_weights=None, group_sym=True, inv_sym=True
    ):
        assert not (kpts_and_weights is None and kpts is None)
        assert (
            isinstance(kpts, tuple)
            or isinstance(kpts, list)
            or isinstance(kpts, np.ndarray)
        )

        self.__group_sym = group_sym
        self.__inv_sym = inv_sym
        self.__mapping_describe = None  # description of k-point mapping to be written from 'calc_kpts_and_weights()' method.
        self.__kpts_mesh = None

        ## Three cases of 'kpts'
        if kpts_and_weights:  # if 'kpts_and_weights' is entered, 'kpts' is ignored.
            assert len(kpts_and_weights) == 2 and len(kpts_and_weights[0]) == len(
                kpts_and_weights[1]
            )
            self.__nkpts = len(kpts_and_weights[0])
            self.__nibzkpts = self.__nkpts
            self.__scaled_kpts, self.__weights = np.array(
                kpts_and_weights[0]
            ), np.array(kpts_and_weights[1])
        elif isinstance(kpts, tuple):
            assert len(kpts) == 3 or len(kpts) == 4
            is_shift = False if kpts[-1] == "gamma" else True
            self.__kpts_mesh = kpts
            self.__nkpts = np.prod(kpts[:3])
            self.__bzkpts, self.__scaled_kpts, self.__weights = self.sampling_kpoints(
                kpts[:3], atoms, group_sym=group_sym, inv_sym=inv_sym, is_shift=is_shift
            )
            self.__nibzkpts = len(self.__scaled_kpts)
        elif isinstance(kpts, list) or isinstance(kpts, np.ndarray):
            self.__nkpts = len(kpts)
            self.__nibzkpts = self.__nkpts
            self.__scaled_kpts, self.__weights = (
                np.array(kpts),
                np.ones(self.__nkpts) / self.__nkpts,
            )
        else:
            assert 0, "Input error."

        ## Unscale the kpts
        self.__kpts = (2 * np.pi) * self.__scaled_kpts @ np.linalg.inv(atoms.get_cell())
        return

    def __str__(self):
        s = str()
        s += "\n============================ [ Kpoint ] ============================="
        s += f"\n* kpts mesh : {self.__kpts_mesh}\n"
        if self.__mapping_describe:
            s += f"* group_sym : {self.__group_sym}\n"
            s += f"* inv_sym   : {self.__inv_sym}\n"
            s += f"* nkpts     : {self.__nkpts}\n"
            s += f"* nibzkpts  : {self.__nibzkpts}\n"
            s += self.__mapping_describe
        else:
            s += f"* nkpts     : {self.__nkpts}\n"
            s += "* Print all k-points"
            for i in range(len(self.__scaled_kpts)):
                s += f"\n {i+1} : {self.__scaled_kpts[i]}"
        s += "\n=====================================================================\n"
        return s

    def sampling_kpoints(
        self,
        kpts_mesh,
        atoms,
        group_sym=True,
        inv_sym=True,
        is_shift=False,
    ):
        """Sampling IBZ k-points using group symmetry.

        :type  kpts_mesh: tuple
        :param kpts_mesh:
            tuple of the number of k-points of each axis, e.g., kpts_mesh=(3, 3, 3)
        :type  atoms: ase.Atoms
        :param atoms:
            Atoms class object
        :type  group_sym: bool
        :param group_sym:
            whether using group symmetry to sample IBZ points
        :type  inv_sym: bool
        :param inv_sym:
            whether using inversion symmetry to sample IBZ points
        :type  is_shift: bool
        :param is_shift:
            whether shifting to the gamma-point, shift to Gamma when is_shift=False
        :rtype: tuple
        :return:
            bzkpts (k-points in BZ), ibzkpts (k-points in IBZ), ibz_w (weights of IBZ k-points)
        """
        bzkpts, ibzkpts, ibz_w = self.group_sym_sampling(
            kpts_mesh, atoms, is_shift
        )  # group symmetry
        if not group_sym:
            if inv_sym:  # inversion symmetry
                ibzkpts, ibz_w = self.inv_sym_sampling(kpts_mesh, bzkpts)
            else:  # no symmetry
                ibzkpts = bzkpts
                ibz_w = np.full(self.__nkpts, 1 / self.__nkpts)
                self.describe_mapping(np.arange(len(bzkpts)), bzkpts, kpts_mesh)
        ibzkpts /= np.array(kpts_mesh, dtype=float)
        return bzkpts, ibzkpts, ibz_w

    def group_sym_sampling(self, kpts_mesh, atoms, is_shift=False):
        is_shift = np.full(3, is_shift)
        is_shift[np.asarray(kpts_mesh) <= 1] = True
        is_shift[np.asarray(kpts_mesh) % 2 == 1] = False

        cell = (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        )
        mapping, bzkpts = spglib.get_ir_reciprocal_mesh(
            kpts_mesh,
            cell,
            is_shift=is_shift,
        )
        bzkpts = bzkpts.astype(float)

        ii = np.array(kpts_mesh) % 2 == 1
        bzkpts[:, ~ii] += 0.5 * is_shift[~ii]

        unique, counts = np.unique(mapping, return_counts=True)
        ibzkpts = bzkpts[unique]
        ibz_w = counts / self.__nkpts
        self.describe_mapping(mapping, bzkpts, kpts_mesh)
        return bzkpts, ibzkpts, ibz_w

    def inv_sym_sampling(self, kpts_mesh, bzkpts):
        mapping = np.arange(len(bzkpts))
        for i in range(len(bzkpts)):
            for j in range(i + 1, len(bzkpts)):
                if mapping[j] != i:
                    is_inv = (abs((bzkpts[i] + bzkpts[j]) % kpts_mesh) < 1e-10).all()
                    if is_inv:
                        mapping[j] = i
        unique, counts = np.unique(mapping, return_counts=True)
        ibzkpts = bzkpts[unique]
        ibz_w = counts / self.__nkpts
        self.describe_mapping(mapping, bzkpts, kpts_mesh)
        return ibzkpts, ibz_w

    def describe_mapping(self, mapping, bzkpts, kpts_mesh):
        self.__mapping_describe = (
            "Print all k-points and mapping to irreducible k-points\n"
        )
        for i, (ir_kpt_id, kpt) in enumerate(zip(mapping, bzkpts)):
            self.__mapping_describe += "%3d ->%3d : %s\n" % (
                i + 1,
                ir_kpt_id + 1,
                kpt.astype(float) / kpts_mesh,
            )
        return

    @property
    def nkpts(self):
        return self.__nkpts

    @property
    def nibzkpts(self):
        return self.__nibzkpts

    @property
    def kpts(self):
        return self.__kpts

    @property
    def weights(self):
        return self.__weights

    @property
    def scaled_kpts(self):
        return self.__scaled_kpts

    def get_kpts_and_weights(self):
        return self.__kpts, self.__weights


def represent_kpts_for_orthorhombic_cell(kpts, atoms, lattice_constants):
    """Represent k-points of some crystal structure for orthorhombic cell's reciprocal vectors.

    :param ndarray kpts: ndarray of (n, 3)
    :param Atoms atoms: ASE Atoms object containing cell
    :param tuple lattice_constants: lattice constants of orthorhombic cell. ex) lattice_constants=(a, b, c)

    :return: ndarray of (n,3). k-points mapped to orthorhombic cell.

    example:
            Represent high-symmetry points of fcc structure for cubic cell.

            >>> from ase.build import bulk
            >>> from gospel.Kpoint import mapping_kpts_to_orthorhombic_cell
            >>> atoms = bulk('Si', 'fcc', a=5.43)
            >>> kpts = atoms.cell.bandpath().kpts
            >>> lattice_constants = (5.43, 5.43, 5.43)
            >>> mapped_kpts = mapping_kpts_to_orthorhombic_cell(kpts, atoms, lattice_constants)
    """
    ## Represent for orthorhombic cell
    ret_kpts = (kpts @ atoms.cell.reciprocal()) * lattice_constants
    return ret_kpts


if __name__ == "__main__":
    print("======================== [ Kpoint Generation Test ] =======================")
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    # atoms = bulk("Al", "fcc", a=3.0, cubic=True)
    # atoms.set_positions(atoms.get_positions() + np.random.randn(*atoms.get_positions().shape))
    print("* atoms =", atoms)
    kpts = (3, 3, 3, "gamma")
    kpts = (3, 3, 3)
    kpts = (2, 2, 2, "gamma")
    print("* kpts =", kpts)

    # group_sym, inv_sym = False, False
    group_sym, inv_sym = False, True
    group_sym, inv_sym = True, True
    kpoint = Kpoint(
        atoms, kpts, kpts_and_weights=None, group_sym=group_sym, inv_sym=inv_sym
    )
    # kpts_and_weights = ([(0,0,0), (0.5,0,0)], [0.5, 0.5])
    # kpoint = Kpoint(atoms, kpts, kpts_and_weights=kpts_and_weights)
    print(kpoint)
    kpts, weights = kpoint.get_kpts_and_weights()

    print("The number of ibz k-points : ", kpoint.nibzkpts)
    print(f"kpoint.kpts=\n{kpoint.kpts}")
    print(f"kpoint.weights=\n{kpoint.weights}")
    print("==========================================================================")
