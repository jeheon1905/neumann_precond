import torch 
import numpy as np
import pylibxc

from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
import os 
from functools import partial

#def target_f(inp, func_name, spin):
#    import pylibxc
#    func = pylibxc.LibXCFunctional(func_name, spin)
#    return func.compute(inp)

class LibXC(metaclass=ABCMeta):
    """https://www.tddft.org/programs/libxc/manual/"""

    spin = 'unpolarized or polarized'
    func = 'LibXCFunctional class object'
    result = 'result of compute method in pylibxc'

    #num_proc = int( os.environ.get('GOSPEL_XC_NUM_PROC', os.environ.get('OMP_NUM_THREADS',1) )  )
    #pool = Pool( int( os.environ.get('OMP_NUM_THREADS', 1)  ) )
    pool = Pool( int( os.environ.get('GOSPEL_XC_NUM_PROC', os.environ.get('OMP_NUM_THREADS',1) )  ) )

    def __init__(self, func_name, spin):
        self.func_name = func_name 
        self.spin      = spin
        self.func = pylibxc.LibXCFunctional(func_name, self.spin)
        #self.run_f     = self.run_f = partial(target_f, func_name=self.func_name, spin=self.spin)
        self.__Vxc = "XC potential"  # will be computed from 'self.compute()'
        self.__Exc = "XC energy"  # will be computed from 'self.compute()'

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def compute_Vxc(self):
        pass

    @abstractmethod
    def compute_Exc(self):
        pass

    def get_describe(self):
        return self.func.describe()

    def parallel_run(self, inp):
        out =  self.func.compute(inp) 
        keys = out.keys()
        result = {}
        for key in keys:
            result[key] = torch.from_numpy( out[key] )
        return result
