import torch
import numpy as np
import pylibxc

from gospel.XC.LibXC import LibXC #, target_f
from gospel.FdOperators import gradient, divergence
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.util import Timer


def calc_sigma(grid, density, spin):
    if spin == "unpolarized":
        #sigma = torch.zeros(1, density.shape[1], dtype=density.dtype, device=density.device)
        dn = gradient(grid, density[0], False)
        sigma = (dn**2).sum(0, keepdim=True)
    else:  # spin == 'polarized'
        sigma = torch.zeros(3, density.shape[1], dtype=density.dtype, device=density.device)
        dn_a = gradient(grid, density[0], False)
        dn_b = gradient(grid, density[1], False)
        sigma[0] = (dn_a**2).sum(0)
        sigma[1] = (dn_a * dn_b).sum(0)
        sigma[2] = (dn_b**2).sum(0)
    return sigma


class GGA(LibXC):
    def __init__(self, func_name, spin="unpolarized"):
        super().__init__(func_name, spin)
        self.__functional_type = "gga"
        return

    @Timer.timeit
    def compute(self, grid, density):
        #st = time.time()
        rho   = density.contiguous().cpu() 
        #et = time.time()
        #PH.synchronize(density.device)
        #print('XC preprocess1 ',et-st )
        #st = time.time()
        sigma = calc_sigma(grid, rho, self.spin)
        #et = time.time()
        #PH.synchronize(density.device)
        #print('XC preprocess2 ',et-st )

        #list_num_points = PH.split_size(rho.size(-1), True)
        inp = {"rho": PH.split(rho).T.contiguous(), "sigma": PH.split(sigma).T.contiguous()} # both tensors should be in cpu
        #inp = {"rho": PH.split(rho.T, dim=0).contiguous(), "sigma": PH.split(sigma.T,dim=0).contiguous()} # both tensors should be in cpu

        #st = time.time()
        result = self.parallel_run( inp)
        #et = time.time()
        #print( 'libXC ', et-st)
        #st = time.time()

        for key, val in result.items():
            result[key] = PH.merge ( val.to(density.device) )
        self._Vxc = self.compute_Vxc(result, grid, density)
        self._Exc = self.compute_Exc(result, grid, density)
        #et = time.time()
        #print('XC postprocess ',et-st )
        return

    def compute_Vxc(self, result, grid, density):
        r"""
        V_{\alpha} = \frac{\partial f}{\partial \rho_{\alpha}}
                     - 2 \nabla \cdot ( \frac{\partial f}{\partial \gamma_{\alpha \alpha}} \nabla \rho_{\alpha} )
                     - \nabla \cdot ( \frac{\partial f}{\partial \gamma_{\alpha \beta}} \nabla \rho_{\beta} )
        """
        vrho = result["vrho"].T
        vsigma = result["vsigma"].T
        assert not torch.any( torch.isnan(vrho) )
        assert not torch.any( torch.isnan(vsigma) )
        Vxc = torch.zeros_like(density)
        if self.spin == "unpolarized":
            dn = gradient(grid, density[0])
            assert not torch.any( torch.isnan(dn) )
            Vxc[0] = vrho[0] - divergence(grid, 2 * vsigma[0] * dn, False )
            assert not torch.any( torch.isnan(Vxc) )
        else:  ## spin == 'polarized'
            dn_a = gradient(grid, density[0])
            dn_b = gradient(grid, density[1])
            tmp = vsigma[1] * dn_b
            Vxc[0] = vrho[0] - divergence(grid, 2 * vsigma[0] * dn_a + tmp, False)
            Vxc[1] = vrho[1] - divergence(grid, 2 * vsigma[2] * dn_a + tmp, False)
        return Vxc

    def compute_Exc(self, result, grid, density):
        r"""
        E_{xc} = \int \epsilon \rho d\vec{r}
        """
        
        return grid.integrate(density.sum(0) * result["zk"].T).item()

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
    from gospel.Grid import Grid
    from ase import Atoms

    
    func_name = "gga_x_pbe"
    spin = "unpolarized"

    atoms = Atoms('H2', positions=[[0,0,0],[0.5,0.0,0]], cell=[1,1,1])
    gpts  = [10,10,10]
    grid = Grid(atoms, gpts, spacing=[1,1,1])
    density = torch.rand(gpts).reshape(1,-1)  # Replace with your actual density

    for func_name in ['gga_c_pbe', 'gga_x_pbe']:
    
        gga = GGA(func_name, spin)
        gga.compute(grid, density)
    
        print(gga.Vxc)
        print(gga.Exc)
