import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn.functional import softmax,silu,sigmoid,leaky_relu,relu


class simple_model(Module):
    """
    This is a simple fmri encoding model
    """
    def __init__(self,_fmaps_fn,Nv,input_shape=(1, 3, 227, 227)):
        """
        Initialization
        :param _fmaps_fn: this is something from st-yves that returns activation map
        :param Nv: Number of voxles
        :param input_shape: size of input images
        """

        #TODO: _fmaps_fn is something that should be done with registering hooks, although it should suffice for now

        super(simple_model, self).__init__()

        self._fmaps_fn = _fmaps_fn
        self.Nv = Nv

        self.device = next(_fmaps_fn.parameters()).device

        self.rf = [] # 3 x window x voxel
        self.w = [] #3 x #channel/layer x voxel
        self.b = Parameter(torch.Tensor(Nv)).float()


        _x = torch.empty((1,) + input_shape[1:], device=self.device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        #styves splits activation maps by resolution so we have an rf per voxel per resolution
        for k,_fm in enumerate(_fmaps): #looping over  layers
            self.rf.append(Parameter(torch.Tensor(_fm.shape[-2],_fm.shape[-1],self.Nv)))
            self.w.append(Parameter(torch.Tensor(_fm.shape[1], self.Nv)))

        self.rf = torch.nn.ParameterList(self.rf)
        self.w = torch.nn.ParameterList(self.w)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(0,len(self.rf)):
            init.kaiming_uniform_(self.w[i], a=math.sqrt(5))
            init.kaiming_uniform_(self.rf[i], a=math.sqrt(5))
        init.uniform_(self.b)

    def forward(self,x):
        _fmaps = self._fmaps_fn(x)
        sigma=torch.zeros(self.Nv).to(self.device)
        for i,fmap in enumerate(_fmaps):
            shp=fmap.shape
            gamma = torch.matmul(fmap.reshape(shp[0], shp[1], shp[-2] * shp[-1]),
                            leaky_relu(self.rf[i].reshape(-1, self.Nv)))#softmax(self.rf[i].reshape(-1, self.Nv),
            # dim=0))#

            gamma = gamma*self.w[i]
            gamma = gamma.sum(dim=1)
            sigma = sigma + gamma
        return sigma+ self.b

class simple_model_terms(Module):
    """
    This is a module that gives all the terms in a simple_model before summation
    """
    def __init__(self,model):
        """
        Initialization
        :param model: a simple_model
        """
        super(simple_model_terms, self).__init__()
        self.model=model

    def forward(self,x,roi_dict):
        """
        Forward function
        :param x: Input image
        :param roi: An roi to filter the output
        :return: the terms in the summation
        """
        _fmaps=self.model._fmaps_fn(x)
        out_dic={}
        for key in roi_dict:
            roi=roi_dict[key]
            print('roi_shape',roi.shape)
            out=[]
            Nv=roi.sum()
            print('NV',Nv)
            for i,fmap in enumerate(_fmaps):
                shp=fmap.shape
                gamma = torch.matmul(fmap.reshape(shp[0], shp[1], shp[-2] * shp[-1]),
                                 leaky_relu(self.model.rf[i][:,:,roi].reshape(-1, Nv)))
                gamma = gamma*self.model.w[i][:,roi]
                out.append(gamma)
            out_dic[key]=torch.cat(out,1)
        return out_dic


