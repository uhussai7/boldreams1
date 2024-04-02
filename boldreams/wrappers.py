import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleDict
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn import ReLU
from torch.nn.functional import softmax,silu,sigmoid,leaky_relu,relu
from torch.nn import Threshold
import numpy as np

class roi_extractor(Module):
    """
    Module to extract one roi
    """
    def __init__(self,roi):
        """
        Initializer
        :param roi: The roi filter of size Nv
        """
        super(roi_extractor, self).__init__()
        self.roi=roi #these are of size visual cortex

    def forward(self,fmri):
        """
        Forward
        :param fmri: Bold signal of size [1,Nv]
        :return: Filtered result
        """
        return fmri[:,self.roi]


class img_extractor(Module):
    """
    Module to extract input img
    """
    def __init__(self,ref_img):
        """
        Initializer
        """
        super(img_extractor, self).__init__()
        self.ref_img=ref_img

    def forward(self,x):
        """
        Forward
        :param fmri: Bold signal of size [1,Nv]
        :return: Filtered result
        """
        return x#self.ref_img

class roi_list_extractor(Module):
    """
    Module to extract from a list of rois
    """
    def __init__(self,roi_dic):
        """
        Initializer
        :param roi_dic: dictonary containing roi names and rois
        """
        super(roi_list_extractor, self).__init__()
        self.roi_dic=roi_dic
        self.extractors={key:roi_extractor(roi=roi_dic[key]) for key in len(self.rois)}

    def forward(self,fmri):
        """
        Forward
        :param fmri:  Bold signal of size [1,Nv]
        :return: List for filtered results
        """
        return [e(fmri) for e in self.extractors.values()]

class dream_wrapper(Module):
    """
    Model wrapper
    """
    def __init__(self,model,roi_dic):
        """
        Initializer
        :param model: Model the predicts the fmri signal
        :param roi_dic: dictonary containing roi names and rois
        """
        super(dream_wrapper, self).__init__()
        self.model=model
        self.roi_dic=roi_dic
        self.roi=ModuleDict({key:roi_extractor(roi=roi_dic[key]) for key in roi_dic})

    def forward(self,x):
        """
        Forward
        :param x: Image batch of shape [B,3,h,w]
        :return: Signal in each roi
        """
        fmri=self.model(x)
        return {e:self.roi[e](fmri) for e in self.roi}#,self.img_e(x)

class dream_wrapper_ref_img(Module):
    """
    Model wrapper
    """
    def __init__(self,model,roi_dic,ref_img):
        """
        Initializer
        :param model: Model the predicts the fmri signal
        :param roi_dic: dictonary containing roi names and rois
        """
        super(dream_wrapper_ref_img, self).__init__()
        self.model=model
        self.roi_dic=roi_dic
        self.roi=ModuleDict({key:roi_extractor(roi=roi_dic[key]) for key in roi_dic})
        self.ref_img=img_extractor(ref_img)

    def forward(self,x):
        """
        Forward
        :param x: Image batch of shape [B,3,h,w]
        :return: Signal in each roi
        """
        fmri=self.model(x)
        return {e:self.roi[e](fmri) for e in self.roi},self.ref_img(x)

class dream_wrapper_terms(dream_wrapper):
    def __init__(self,model,roi_dic):
        super(dream_wrapper_terms, self).__init__(model,roi_dic)
        self.roi=ModuleDict({key:roi_extractor_terms(roi=roi_dic[key]) for key in roi_dic})

class roi_extractor_terms(roi_extractor):
    def __init__(self,roi):
        super(roi_extractor_terms, self).__init__(roi)
    def forward(self,fmri):
        return fmri[:,:, self.roi]

class clip_wrapper(Module):
    def __init__(self,clip,roi_encoder,text):
        super(clip_wrapper, self).__init__()
        self.model=clip
        self.encoder=roi_encoder
        self.text=text
    def forward(self,img):
        return self.model(img, self.text.cuda())[0][0].unsqueeze(0),self.encoder(img)