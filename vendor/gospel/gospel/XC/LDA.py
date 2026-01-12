import torch
import pylibxc

from gospel.XC.LibXC import LibXC
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.util import Timer


class LDA(LibXC):
    def __init__(self, func_name, spin="unpolarized"):
        super().__init__(func_name, spin)
        self.__functional_type = "lda"
        return

    @Timer.timeit
    def compute(self, grid, density):
        import time 

        st= time.time()
        inp = {"rho": PH.split(density).T.contiguous().cpu()}
        PH.synchronize(density.device)
        et= time.time()
        print('to cpu: ', et-st)
        st= time.time()
        result = self.parallel_run( inp)
        PH.synchronize(density.device)
        et= time.time()
        print('libxc : ', et-st)
        st= time.time()
        for key, val in result.items():
            result[key] = PH.merge(val.to(density.device) )
        PH.synchronize(density.device)
        et= time.time()
        print('merge : ', et-st)

#        inp = {"rho": density.T.contiguous().cpu()}
#        result = self.func.compute(inp)
#        for key, val in result.items():
#            result[key] = torch.from_numpy(val).to(density.device, non_blocking=True)

        self._Vxc = self.compute_Vxc(result).to(density.device)
        self._Exc = self.compute_Exc(result, density) * grid.microvolume
        return

    def compute_Vxc(self, result):
        r"""
        V_{xc, \alpha} = \frac{\partial f}{\partial \rho_{\alpha}}
                       = vrho_{\alpha}
        """
        return result["vrho"].T

    def compute_Exc(self, result, density):
        r"""
        E_{xc} = \int \epsilon \rho d\vec{r}
        """
        # depending on version of libxc, shape of result['zk'] varies
        # therefore, flatten result and density and perform dot product  Sunghwan Choi
        return (sum(density).reshape(-1) @ result["zk"].reshape(-1) ).item()

    @property
    def Vxc(self):
        return self._Vxc

    @property
    def Exc(self):
        return self._Exc

    @property
    def functional_type(self):
        return self.__functional_type


if __name__ == "__main__":
    import torch
    from time import time
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_try", help="number of retry",  type=int, default=5)
    parser.add_argument("--n_grid", help="number of grid point",  type=int, default=1000)
    parser.add_argument("--device", help="device to comput lda functional",  type=str, default="cpu")
    args = parser.parse_args()

    func_name = "lda_x"
    print("func_name = ", func_name)
    spin = "unpolarized"
    lda = LDA(func_name, spin)

    class Grid:
        microvolume = 1.0

    grid = Grid()

    device = torch.device(f"cuda:{PH.rank}" if args.device=="cuda" else "cpu" )

    if spin == "unpolarized":
        density = torch.zeros((1, args.n_grid), device=device)
    else:
        density = torch.zeros((2, args.n_grid), device=device)
    density[0] = torch.arange(0.1, 0.6, 0.5/args.n_grid)

    print("density = \n", density)
    density = density.double() # type casting 
    st = time()
    for _ in range(args.n_try):
        ref_Vxc = -((3 / torch.pi) ** (1 / 3)) * density ** (1 / 3)
    et = time()
    print((et-st) /args.n_try, "seconds")
    print("ref_Vxc = ", ref_Vxc)

    st = time()
    for _ in range(args.n_try):
        lda.compute(grid, density)
    et = time()
    print((et-st) /args.n_try, "seconds")
    print(f"Vxc = {lda.Vxc}")
    print(f"Exc = {lda.Exc}")
