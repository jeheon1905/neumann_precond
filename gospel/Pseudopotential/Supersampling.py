import scipy 
import numpy as np
try:
    import torch 
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Supersampling requires pytorch. please install pytorch first and retry")

def run(conv, data, pbc):
    data = torch.from_numpy(data)
    size = data.size()
    data = data.repeat(3,3,3) # padding 
    data = data[size[0] - (conv.conv.weight.size(-3)-1)//2: 2*size[0] + (conv.conv.weight.size(-3)-1)//2,
                size[1] - (conv.conv.weight.size(-2)-1)//2: 2*size[1] + (conv.conv.weight.size(-2)-1)//2,     
                size[2] - (conv.conv.weight.size(-1)-1)//2: 2*size[2] + (conv.conv.weight.size(-1)-1)//2,]
    conv = conv.type(data.dtype)
    with torch.no_grad():
        data = conv(data.unsqueeze(0).unsqueeze(0))
    return data.squeeze(0).squeeze(0)
    #return data.squeeze(0).squeeze(0).numpy()


class TruncSincSupersampling3D(torch.nn.Module):
    def __init__(self, fine, limit, original_spacing):
        super(TruncSincSupersampling3D, self).__init__()
        assert len(fine)==3 and  len(limit)==3 and len(original_spacing)==3, str(len(fine)) +" " +str(len(limit)) +" " +str(len(original_spacing)) 
        

        kernel_size = 2*np.array(limit) * np.array(fine)
        microvolume  = original_spacing[0]*original_spacing[1]*original_spacing[2]/float(fine[0]*fine[1]*fine[2]) # microvolume
        analytic_integral = 8/np.power(np.pi,3)*original_spacing[0]*original_spacing[1]*original_spacing[2]*scipy.special.sici(limit[0]*np.pi)[0]*scipy.special.sici(limit[1]*np.pi)[0]*scipy.special.sici(limit[2]*np.pi)[0]

        f = lambda x: np.sinc( x[0]/original_spacing[0])*np.sinc( x[1]/original_spacing[1]) * np.sinc( x[2]/original_spacing[2])

        weight = f( np.meshgrid(np.mgrid[-limit[0]*original_spacing[0]:limit[0]*original_spacing[0]:1j*(kernel_size[0]+1) ],
                                np.mgrid[-limit[1]*original_spacing[1]:limit[1]*original_spacing[1]:1j*(kernel_size[1]+1) ],
                                np.mgrid[-limit[2]*original_spacing[2]:limit[2]*original_spacing[2]:1j*(kernel_size[2]+1) ] ) )
        #print(kernel_size)
        #print(weight[0] , weight[1], weight[2])
        self.conv = torch.nn.Conv3d(1, 1, kernel_size, fine, bias=False)
        self.conv.weight = torch.nn.Parameter( torch.from_numpy( weight*microvolume / analytic_integral ).unsqueeze(0).unsqueeze(0)  )
    def forward(self, data) :
        return self.conv(data)
