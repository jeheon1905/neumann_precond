import math
import torch
from .GGA import GGA, calc_sigma
from gospel.ParallelHelper import ParallelHelper as PH
from gospel.util import timer


param =  {'beta':0.06672455060314922, 'gamma': 0.031090690869654895034, 'BB': 1.0, 
          'kappa': 0.8040, 'mu':0.2195149727645171,
          'dens_threshold': 1.000e-15, 'sigma_threshold': 2.073e-317, 'zeta_threshold': 1.000e-20,
         }

class PBE_c(GGA):
    def __init__(self, func_name, spin):
        assert spin=='unpolarized'
        assert func_name=='gga_c_pbe'
        super().__init__(func_name, spin)
        self.__functional_type ='gga'
        #self.e_func = torch.compile(pbe_ec_unpol)
        #self.v_func = torch.compile(pbe_vc_unpol)

        return 
    @timer
    def test(self, grid, density):
        from .PBE import param
        sigma = calc_sigma(grid, density, self.spin)

        index = density<param['dens_threshold']
        zk =  pbe_ec_unpol(param, density, sigma)
        _, vrho, vsigma = pbe_vc_unpol(param, density, sigma)

        zk[index]=0.0
        vrho[index]=0.0
        vsigma[index]=0.0

        self.compute_Vxc({'vrho':vrho.T, 'vsigma':vsigma.T}, grid, density)
        self.compute_Exc({'zk':zk.T}, grid, density)

        return 
    @timer
    def compute(self, grid, density):
        from .PBE import param
        sigma = calc_sigma(grid, density, self.spin)

#        __density = PH.split(density)
#        __sigma   = PH.split(sigma) 

        index = density<param['dens_threshold']
        zk =  pbe_ec_unpol(param, density, sigma)
        _, vrho, vsigma = pbe_vc_unpol(param, density, sigma)

        zk[index]=0.0
        vrho[index]=0.0
        vsigma[index]=0.0
        
#        zk    = PH.merge(zk)
#        vrho  = PH.merge(vrho)
#        vsigma= PH.merge(vsigma)

        self._Vxc = self.compute_Vxc({'vrho':vrho.T, 'vsigma':vsigma.T}, grid, density)
        self._Exc = self.compute_Exc({'zk':zk.T}, grid, density)
        return 

class PBE_x(GGA):
    def __init__(self, func_name, spin):
        assert spin=='unpolarized'
        assert func_name=='gga_x_pbe'
        super().__init__(func_name, spin)
        self.__functional_type ='gga'
        #self.e_func = torch.compile(pbe_ex_unpol)
        #self.v_func = torch.compile(pbe_vx_unpol)
        return 

    @timer
    def test(self, grid, density):
        from .PBE import param
        sigma = calc_sigma(grid, density, self.spin)
        zk =  pbe_ex_unpol(param, density, sigma)
        _, vrho, vsigma = pbe_vx_unpol(param, density, sigma)
        return 

    @timer
    def compute(self, grid, density):
        from .PBE import param
        sigma = calc_sigma(grid, density, self.spin)

        index = density<param['dens_threshold']
        zk =  pbe_ex_unpol(param, density, sigma)
        _, vrho, vsigma = pbe_vx_unpol(param, density, sigma)

        zk[index]=0.0
        vrho[index]=0.0
        vsigma[index]=0.0

        self._Vxc = self.compute_Vxc({'vrho':vrho.T, 'vsigma':vsigma.T}, grid, density)
        self._Exc = self.compute_Exc({'zk':zk.T}, grid, density)
        return 
    
def my_piecewise3(c, x1, x2):
    if type(c)==type(True):
        return x1 if c else x2 
    else:
        return torch.where(c, x1, x2)

def my_piecewise5(c1, x1, c2, x2, x3):
    if type(c1)==type(True):
        return x1 if c1 else (x2 if c2 else x3) 
    else:
        return torch.where(c1, x1, torch.where(c2, x2, x3))

#@torch.compile    
def pbe_ec_unpol(param, rho, sigma):
    t1 = 1.442249570307408382321638310780109588392
    t2 = 1 / torch.pi
    t3 = math.pow(t2, 1.0/3.0)
    t4 = t1*t3
    t5 = 1.587401051968199474751705639272308260391
    t6 = t5*t5
    t7 = torch.pow(rho[0], 1.0/3.0)    
    
    t10 = t4*t6 /t7
    t12 = 1 + 0.53425e-1 * t10
    t13 = torch.sqrt(t10)
    t16 = torch.pow(t10, 3.0/2.0)
    t18 = t1 *t1
    t19 = t3*t3
    t20 = t18*t19
    t21 = t7*t7
    t24 = t20*t5 / t21
    t26 = 0.379785e1 * t13 + 0.8969e0 * t10 + 0.204775e0 * t16 + 0.123235e0 * t24
    t29 = 0.1e1 + 0.16081979498692535067e2 / t26
    del t26, 
    t30 = torch.log(t29)
    del t29
    t32 = 0.621814e-1 * t12 * t30
    del t12, t30
    t33 = 0.1e1 <= param['zeta_threshold']

    t34 = math.pow(param['zeta_threshold'], 1.0/3.0)
    t36 = my_piecewise3(t33, t34 * param['zeta_threshold'], 1)
    t39 = 1.259921049894873164767210607278228350570
    t43 = (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1)
    t45 = 0.1e1 + 0.278125e-1 * t10
    t50 = 0.51785e1 * t13 + 0.905775e0 * t10 + 0.1100325e0 * t16 + 0.1241775e0 * t24
    del t10, t13, t16, t24
    t53 = 0.1e1 + 0.29608749977793437516e2 / t50
    del t50
    t54 = torch.log(t53)
    del t53
    t57 = 0.19751673498613801407e-1 * t43 * t45 * t54
    del t45,t54
    t58 = t34 * t34
    t59 = my_piecewise3(t33, t58, 1)
    t60 = t59 * t59
    t61 = t60 * t59
    t62 = param['gamma'] * t61
    t63 = rho[0] * rho[0]
    t65 = 0.1e1 / t7 / t63

    del t7

    t68 = 0.1e1 / t60
    t70 = 0.1e1 / t3
    t72 = t68 * t18 * t70 * t5
    t75 = param['BB'] * param['beta']
    t76 = 0.1e1 / param['gamma']
    t79 = 0.1e1 / t61
    t81 = torch.exp(-(-t32 + t57) * t76 * t79)
    t82 = t81 - 0.1e1
    del t81
    t83 = 0.1e1 / t82
    del t82
    t84 = t76 * t83
    t85 = sigma[0] * sigma[0]
    t87 = t75 * t84 * t85
    del t84, t85
    t88 = t63 * t63
    del t63
    t90 = 0.1e1 / t21 / t88
    del t21, t88
    t91 = t39 * t39
    t92 = t90 * t91
    del t90
    t93 = t60 * t60
    t94 = 0.1e1 / t93
    t95 = t92 * t94
    del t92
    t96 = 0.1e1 / t19
    t97 = t1 * t96
    t98 = t97 * t6
    t99 = t95 * t98
    del t95
    t102 = sigma[0] * t65 * t39 * t72 / 0.96e2 + t87 * t99 / 0.3072e4
    del t65, t87, t99
    t103 = param['beta'] * t102
    t104 = param['beta'] * t76
    t107 = t104 * t83 * t102 + 0.1e1
    del t83, t102
    t108 = 0.1e1 / t107
    del t107
    t109 = t76 * t108
    del t108
    t111 = t103 * t109 + 0.1e1
    del t103, t109
    t112 = torch.log(t111)
    del t111
    t113 = t62 * t112
    del t112

    return  (-t32 + t57 + t113).reshape(1,-1)


#@torch.compile    
def pbe_vc_unpol(param, rho, sigma):
    t1 = 1.442249570307408382321638310780109588392
    t2 = 1 / torch.pi
    t3 = math.pow( t2, 1.0/3.0)
    t4 = t1* t3 
    t5 = 1.587401051968199474751705639272308260391
    t6 = t5*t5
    t7 = torch.pow(rho[0], 1.0/3.0)
    t10 = t4 * t6 /t7
    print(t4, t6, t7)
    print(t10)
    t12 = 1 + 0.53425e-1 * t10
    t13 = torch.sqrt(t10)
    t16 = torch.pow(t10, 3.0 / 2.0)
    t18 = t1 * t1
    t19 = t3 * t3
    t20 = t18 * t19 
    t21 = t7*t7
    t24 = t20* t5 / t21
    t26 = 0.379785e1 * t13 + 0.8969e0 * t10 + 0.204775e0 * t16 + 0.123235e0 * t24
    t29 = 1 + 0.16081979498692535067e2 / t26
    t30 = torch.log(t29)
    print(t30)
    t32 = 0.621814e-1 * t12 * t30
    t33 = 1<=param['zeta_threshold']
    t34 = math.pow(param['zeta_threshold'], 1.0/3.0)
    t36 = my_piecewise3(t33, t34 * param['zeta_threshold'], 1)
    t39 = 1.259921049894873164767210607278228350570
    t43 = (2 * t36 - 2) / (2 * t39 - 2)
    t45 = 1 + 0.278125e-1 * t10
    t50 = 0.51785e1 * t13 + 0.905775e0 * t10 + 0.1100325e0 * t16 + 0.1241775e0 * t24
    del t16, t24
    t53 = 1 + 0.29608749977793437516e2 / t50
    t54 = torch.log(t53)
    t57 = 0.19751673498613801407e-1 * t43 * t45 * t54
    t58 = t34 * t34
    t59 = my_piecewise3(t33, t58, 1)
    t60 = t59 * t59
    t61 = t60 * t59
    t62 = param['gamma'] *t61
    t63 = rho[0] * rho[0]
    t65 = 1 / t7 / t63
    t68 = 1 / t60
    t70 = 1 / t3
    t72 = t68 * t18 * t70 * t5
    t75 = param['BB'] * param['beta']
    t76 = 1.0 / param['gamma']
    t79 = 1.0 / t61
    t81 = torch.exp(-(-t32 + t57) * t76 * t79)
    t82 = t81 - 1.0
    t83 = 1.0 / t82
    t84 = t76 * t83
    t85 = sigma[0]*sigma[0]
    t87 = t75 * t84 * t85
    t88 = t63 * t63
    t90 = 1.0 / t21 / t88
    t91 = t39 * t39
    t92 = t90 * t91
    t93 = t60 * t60
    t94 = 1.0 / t93
    t95 = t92 * t94
    del t92
    t96 = 1.0 / t19
    t97 = t1 * t96
    t98 = t97 * t6
    t99 = t95 * t98
    del t95
    t102 = sigma[0] * t65 * t39 * t72 / 0.96e2 + t87 * t99 / 0.3072e4
    t103 = param['beta'] * t102
    t104 = param['beta'] * t76
    t107 = t104 * t83 * t102  + 1
    t108 = 1.0 / t107
    t109 = t76 * t108
    del t108
    t111 = t103 * t109 + 1
    t112 = torch.log(t111)
    t113 = t62 * t112
    del t112 
    out = {'zk' : (-t32 + t57 + t113).reshape(1, -1) }

    t115 = 1.0 / t7 / (rho[0] + 0.000020)
    t116 = t6 * t115
    #print(t7, "\n", rho[0], '\n', t115,"\n", t6)
    t118 = t4 * t116 * t30
    del t30
    t119 = 0.11073470983333333333e-2 * t118
    del t118
    t120 = t26 * t26
    del t26
    t121 = 0.1e1 / t120
    del t120
    t122 = t12 * t121
    del t12, t121
    t124 = 0.1e1 / t13 * t1
    del t13
    t125 = t3 * t6
    t126 = t125 * t115
    t127 = t124 * t126
    del t124
    t129 = t4 * t116
    del t116
    t131 = torch.sqrt(t10)
    del t10
    t132 = t131 * t1
    del t131
    t133 = t132 * t126
    del t126, t132
    t138 = t20 * t5 / t21 / rho[0]
    t140 = -0.632975e0 * t127 - 0.29896666666666666667e0 * t129 - 0.1023875e0 * t133 - 0.82156666666666666667e-1 * t138
    t141 = 0.1e1 / t29
    del t29
    t142 = t140 * t141
    del t140, t141
    t143 = t122 * t142
    del t122, t142
    t144 = 0.1e1 * t143
    del t143
    t145 = t43 * t1
    t148 = t145 * t125 * t115 * t54
    del t54, t115
    t149 = 0.18311447306006545054e-3 * t148
    del t148
    t150 = t43 * t45
    del t45
    t151 = t50 * t50 
    del t50
    t152 = 0.1e1 / t151
    del t151
    t157 = -0.86308333333333333334e0 * t127 - 0.301925e0 * t129 - 0.5501625e-1 * t133 - 0.82785e-1 * t138
    del t127, t129, t133, t138
    t159 = 0.1e1 / t53 
    del t53
    t160 = t152 * t157 * t159
    del t152, t157, t159
    t161 = t150 * t160
    del t150, t160
    t162 = 0.5848223622634646207e0 * t161
    del t161
    t163 = t63 * rho[0]
    del t63
    t165 = 0.1e1 / t7 / t163
    del t7, t163
    t170 = param['gamma'] * param['gamma']
    t171 = 0.1e1 / t170
    t172 = t75 * t171
    t173 = t82 * t82
    del t82
    t174 = 0.1e1 / t173
    del t173
    t175 = t174 * t85
    del t85
    t176 = t175 * t90
    del t90,t175
    t177 = t172 * t176
    del t176
    t179 = 0.1e1 / t93 / t61
    t180 = t91 * t179
    t181 = t180 * t1
    t182 = t96 * t6
    t183 = t119 + t144 - t149 - t162
    t184 = t183 * t81
    t185 = t182 * t184
    del t184
    t186 = t181 * t185
    del t185
    t189 = t88 * rho[0]
    del t88
    t191 = 0.1e1 / t21 / t189
    del t21, t189
    t192 = t191 * t91
    del t191
    t193 = t192 * t94
    del t192
    t194 = t193 * t98
    del t193
    t197 = -0.7e1 / 0.288e3 * sigma[0] * t165 * t39 * t72 + t177 * t186 / 0.3072e4 - 0.7e1 / 0.4608e4 * t87 * t194
    del t87, t165, t177, t186, t194
    t198 = param['beta'] * t197
    t200 = t107 * t107
    del t107
    t201 = 0.1e1 / t200
    del t200
    t202 = t76 * t201
    t204 = param['beta'] * t171 * t174
    del t174
    t206 = t79 * t81
    del t81
    t211 = t204 * t102 * t183 * t206 + t104 * t83 * t197
    del t183, t197, t204, t206
    t212 = t202 * t211
    del t202, t211
    t214 = -t103 * t212 + t198 * t109
    del t103, t198, t212
    t215 = 0.1e1 / t111
    del t111
    t217 = t62 * t214 * t215
    del t214
    out['vrho'] = (-t32 + t57 + t113 + rho[0] * (t119 + t144 - t149 - t162 + t217) ).reshape(1,-1)
    del t32, t57,t119, t144, t149, t162, t217
    t220 = rho[0] * param['gamma']
    t224 = t18 * t70 * t5
    t228 = t75 * t84 * sigma[0]
    del t84
    t231 = t65 * t39 * t68 * t224 / 0.96e2 + t228 * t99 / 0.1536e4
    del t65, t99, t224, t228
    t232 = param['beta'] * t231
    t234 = param['beta'] * param['beta']
    t235 = t234 * t102
    del t102
    t236 = t235 * t171
    del t235
    t237 = t201 * t83
    del t83,t201
    t238 = t237 * t231
    del t231, t237
    t240 = t232 * t109 - t236 * t238
    del t109, t232, t236, t238
    out['vsigma'] =  (t220 * t61 * t240 * t215).reshape(1,-1)
    del t215, t220, t240
  
    return out['zk'], out['vrho'] , out['vsigma']



#@torch.compile    
def pbe_ex_unpol(param, rho, sigma):

    t2 = rho[0] / 2 <= param['dens_threshold']
    t3 = 1.442249570307408382321638310780109588392
    t4 = 1.464591887561523263020142527263790391739
    t6 = t3 / t4
    t7 = 1.0 <= param['zeta_threshold']
    t8 = param['zeta_threshold'] - 1.0
    t10 = my_piecewise5(t7, t8, t7, -t8, 0)
    t11 = 1.0 + t10
    t13 = math.pow(param['zeta_threshold'], 1.0 / 3.0)
    t15 = math.pow(t11, 1.0 / 3.0)
    t17 = my_piecewise3(t11 <= param['zeta_threshold'], t13 * param['zeta_threshold'], t15 * t11)
    t18 = torch.pow(rho[0], 1.0 / 3.0)
    t20 = 1.817120592832139658891211756327260502428
    t22 = math.pow(torch.pi, 2.0)
    t23 = math.pow(t22, 1.0 / 3.0)
    t24 = t23 * t23
    t25 = 1.0 / t24
    t27 = 1.259921049894873164767210607278228350570
    t28 = t27 * t27
    t30 = rho[0] * rho[0]
    t31 = t18 * t18
    t33 = 1.0 / t31 / t30
    del t30, t31

    t37 = param['kappa'] + param['mu'] * t20 * t25 * sigma[0] * t28 * t33 / 0.24e2
    del t33
    t42 = 1.0 + param['kappa'] * (1.0 - param['kappa'] / t37)
    del t37
    t46 = my_piecewise3(t2, 0, -3 / 8 * t6 * t17 * t18 * t42)
    del t18, t42
    tzk0 = 2.0 * t46
    del t46
    return tzk0.reshape(1,-1)



#@torch.compile    
def pbe_vx_unpol(param, rho, sigma):
    #params = param['params']

    t2 = rho[0] / 2 <= param['dens_threshold']
    t3 = 1.442249570307408382321638310780109588392
    t4 = 1.464591887561523263020142527263790391739
    t6 = t3 / t4
    t7 = 1.0 <= param['zeta_threshold']
    t8 = param['zeta_threshold'] - 1.0
    t10 = my_piecewise5(t7, t8, t7, -t8, 0)
    t11 = 1.0 + t10
    t13 = math.pow(param['zeta_threshold'], 1.0 / 3.0)
    t15 = math.pow(t11, 1.0 / 3.0)
    t17 = my_piecewise3(t11 <= param['zeta_threshold'], t13 * param['zeta_threshold'], t15 * t11)
    t18 = torch.pow(rho[0], 1.0 / 3.0)
    t20 = 1.817120592832139658891211756327260502428
    t22 = torch.pi*torch.pi
    t25 = math.pow(t22, -2.0 / 3.0)
    t27 = 1.259921049894873164767210607278228350570
    t28 = t27 * t27
    t30 = rho[0] * rho[0]
    t31 = t18 * t18 
    t33 = 1.0 / t31 / t30
    t37 = param['kappa'] + param['mu'] * t20 * t25 * sigma[0] * t28 * t33 / 24
    del t33
    t42 = 1.0 + param['kappa'] * (1.0 - param['kappa'] / t37)
    t46 = my_piecewise3(t2, 0, -3 / 8 * t6 * t17 * t18 * t42)
    tzk0 = 2 * t46

    out = {'zk': tzk0.reshape(1,-1)}


    t52 = t30 * rho[0]
    t56 = param['kappa'] * param['kappa']
    t58 = t6 * t17 / t18 / t52 * t56
    del t52
    t59 = t37 * t37
    del t37
    t61 = 1.0 / t59 * param['mu']
    del t59
    t64 = t25 * sigma[0] * t28
    t65 = t61 * t20 * t64
    del t64
    t69 = my_piecewise3(t2, 0, -t6 * t17 / t31 * t42 / 8 + t58 * t65 / 24)
    del t31, t42, t58, t65
    tvrho0 = 2.0 * rho[0] * t69 + 2.0 * t46
    del t46, t69
    out['vrho'] = tvrho0.reshape(1,-1)

    t78 = t20 * t25 * t28
    t79 = t61 * t78
    del t61, t78
    t82 = my_piecewise3(t2, 0, -t6 * t17 / t18 / t30 * t56 * t79 / 64)
    del t18, t30, t79
    tvsigma0 = 2.0 * rho[0] * t82
    del t82
    out['vsigma'] = tvsigma0.reshape(1,-1)

    return out['zk'], out['vrho'], out['vsigma']

if __name__ =="__main__":
    from ase import Atoms 
    from gospel.Grid import Grid
    from gospel.FdOperators import gradient, divergence
    from gospel.XC.GGA import GGA

    #func_name = "gga_x_pbe"
    spin = "unpolarized"

    atoms = Atoms('H2', positions=[[0,0,0],[0.5,0.0,0]], cell=[1,1,1])
    gpts  = [10,10,10]
    grid = Grid(atoms, gpts, spacing=[1,1,1])
    density = torch.rand(gpts, dtype=torch.double).reshape(1,-1)  # Replace with your actual density
    sigma = calc_sigma(grid, density, spin)

    #print( pbe_ec_unpol(param, density, sigma) )
    #print( pbe_ex_unpol(param, density, sigma) )
    #_, vrho, vsigma =  pbe_vx_unpol(param, density, sigma) 
    #_, vrho, vsigma =  pbe_vc_unpol(param, density, sigma) 
    #print(_)
    #print(vrho)
    #print(vsigma)

    dn = gradient(grid, density[0])
    print( vrho[0] -divergence(grid, 2*vsigma[0]*dn, False) )
    #for func_name in ['gga_c_pbe', 'gga_x_pbe']:
    for func_name in [ 'gga_c_pbe']:
    
        gga = GGA(func_name, spin)
        gga.compute(grid, density)
    
        print(gga.Vxc)
        #print(gga.Exc)
