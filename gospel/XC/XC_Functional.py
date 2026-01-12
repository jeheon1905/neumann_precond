import numpy as np
import torch 
import re
from gospel.util import timer
from gospel.Density import Density
from gospel.ParallelHelper import ParallelHelper as PH
from .LDA import LDA
from .GGA import GGA
#from .KLI import KLI
from .Slater import KLI
from .PBE import PBE_c, PBE_x # speical function 

lda_type_list    = ['lda_x', 'lda_c_pz']
gga_type_list    = ['gga_x_pbe', 'gga_c_pbe']
mgga_type_list   = []
exx_type_list    = ['slater', 'exx']


class XC_Functional:
    """
    func_info   :   dictionary type
    func_list   :   list of various functionals
    [Input]
        func_name  : str type
        weight     : str type
        spin       : str type ('polarized' or 'unpolarized')
    """
    def __init__(self, func_name='gga_x_pbe + gga_c_pbe', weight=None, calc=None):
        self.__calc = calc
        self.__spin = calc.parameters.spin
        self.__kpts = calc.kpoint.kpts
        self.__grid = calc.grid

        ## Input processing
        self.func_info = self._str_to_dict_xc(func_name.lower(), weight)

        ## Clasify the type of functional
        self.func_list = self._clasify_func_type(self.func_info)
        return


    def _str_to_dict_xc(self, func_name, weight):
        """
        Check whether the input parameters are valid and
        make dictionary type object that can be used in 'compute' method.
        (Usage example)
        >>> functional = XC_Functional(func_name="gga_x_pbe+gga_c_pbe", weight="1.0, 1.0")
        >>> print( functional.get_functional_list() )
        {'functional': ['gga_x_pbe', 'gga_c_pbe'], 'weight': array([1., 1.])}

        >>> functional = XC_Functional(func_name="gga_x_pbe+gga_c_pbe")
        >>> print( functional.get_functional_list() )
        {'functional': ['gga_x_pbe', 'gga_c_pbe'], 'weight': array([1., 1.])}

        >>> functional = XC_Functional()
        >>> print( functional.get_functional_list() )
        {'functional': [], 'weight': array([], dtype=float64)}
        """
        func_info = {}

        func_name_list = np.asarray(list(filter(None, re.split('[+ ,]', func_name))))

        weight_list = np.array([])
        if weight is None:
            weight_list = np.ones(len(func_name_list))
        else:
            weight_list = np.asarray(list(map(float, filter(None, re.split('[, ]', weight)))))
            assert len(func_name_list) == len(weight_list), "The number of functional names and weights must be same."

        func_info['functional'] = func_name_list
        func_info['weight'] = weight_list
        return func_info


    def _clasify_func_type(self, func_info):
        """
        Clasify the functional type of func_info and
        return the corresponding class object.
        """
        func_list = []
        for func_name in func_info['functional']:
            if func_name=='gga_x_pbe' and self.__spin=="unpolarized":
                func_list.append( PBE_x(func_name, self.__spin) )
                #func_list[-1].test(self.__grid, torch.rand((1,self.__grid.ngpts) , device=PH.get_device(self.__calc.parameters['use_cuda']), dtype=torch.double) )  # test run 
            elif func_name =='gga_c_pbe' and self.__spin=="unpolarized":
                func_list.append( PBE_c(func_name, self.__spin) )
                #func_list[-1].test(self.__grid, torch.rand((1,self.__grid.ngpts) , device=PH.get_device(self.__calc.parameters['use_cuda']), dtype=torch.double) )  # test run 
            elif func_name in lda_type_list:
                func_list.append( LDA(func_name, self.__spin) )
            elif func_name in gga_type_list:
                func_list.append( GGA(func_name, self.__spin) )
            elif func_name in mgga_type_list:
                func_list.append( MGGA(func_name, self.__spin) )
            elif func_name in exx_type_list:
                #func_list.append( KLI(self.__grid, self.__kpts) )
                func_list.append( KLI(self.__grid, self.__kpts, func_name) )
            elif func_name == 'None':
                continue
            else:
                assert 0, str(func_name)+" is not available functional. Check the input name of functional."
        return func_list


    @timer
    def compute(self, density):
        """
        Arguments:
            density (Density or np.ndarray) : gospel.Density class object or np.ndarray
        """
        for func, weight in zip(self.func_list, self.func_info['weight']):
            #if func.functional_type == 'exx':
            if func.functional_type in exx_type_list:
                assert isinstance(density, Density)
                func.compute(density.get_density(nlcc=True),
                             density.eigvec, density.occ)
            else:
                func.compute(self.__grid, density if isinstance(density, np.ndarray) or isinstance(density, torch.Tensor)
                                                  else density.get_density(nlcc=True))
        return


    def get_Vxc(self):
        Vxc = None
        for func, weight in zip(self.func_list, self.func_info['weight']):
            Vxc = (weight * func.Vxc) if Vxc is None else (Vxc +weight * func.Vxc)
            #Vxc.append(weight * func.Vxc)
        #Vxc = np.sum(Vxc, axis=0)
        return Vxc


    def get_Exc(self):
        Exc = 0.0
        for func, weight in zip(self.func_list, self.func_info['weight']):
            Exc += weight * func.Exc
        return Exc


    def get_functional_list(self):
        return self.func_info


    def get_describe(self):
        return [ self.func_list[i].get_describe() for i in range(len(self.func_list)) ]


    def __str__(self):
        s = str()
        s += '\n========================== [ XC_Functional ] ==========================\n'
        s += f"* functional : {self.func_info['functional']}\n"
        s += f"* weight     : {self.func_info['weight']}\n"
        for i in range(len(self.func_list)):
            s += f"\n{self.func_list[i].get_describe()}"
        s += '\n=======================================================================\n'
        return s
