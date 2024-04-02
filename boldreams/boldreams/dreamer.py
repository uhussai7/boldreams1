#we want a dreamer class, something that set up all the transforms and stuff
#1) input will be the model and ROI
#2) optional inputs will be needed for ref_image, type of objective, xfm, etc.

from torchvision.transforms import RandomAffine
from torch.optim import Adam,SGD
from boldreams.objectives import *
from boldreams.wrappers import *
from boldreams.param import ref_image

class prep_dreamer:
    def __init__(self,encoder,rois,input_size,ref_img=None):
        '''
        Init
        :param encoder: the trained encoder
        :param rois: the dictionary for ROIs
        '''
        if ref_img!=None:
            self.dreamer=dream_wrapper_ref_img(encoder,rois,ref_img)
            self.set_input(ref_img)
        else:
            self.dreamer=dream_wrapper(encoder,rois)
            self.set_input()

        self.rois=rois
        self.input_size=input_size

        self.set_xfm()
        self.set_optim()
        self.set_objective('roi',"roi_"+list(rois.keys())[0])

    def set_objective(self,obj_name,*args,**kwargs):
        self.obj_name=obj_name
        self.obj=globals().get(obj_name)(*args,**kwargs)

    def set_xfm(self,xfm=None):
        if xfm==None:
            self.xfm=[RandomAffine(8,translate=[0.05,0.05],scale=[0.5,0.9], fill=0.0)]
        else:
            self.xfm=xfm

    def set_optim(self,optim=None,lr=5e-3):
        if optim==None:
            self.optim=lambda params: Adam(params,lr=lr)
        else:
            self.optim=lambda params: optim(params,lr=lr)

    def set_input(self,ref_img=None,batch=1):
        if ref_img==None:
            self.param_f=lambda: param.image(self.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=batch)
        else:
            self.param_f = lambda: ref_image(torch.clone(ref_img))

    def dream(self,iterations=1024):
        _ = render.render_vis(self.dreamer.cuda().eval(), self.obj, param_f=self.param_f, transforms=self.xfm,
                              optimizer=self.optim,
                              fixed_image_size=self.input_size[-1],
                              thresholds=(int(iterations),),
                              show_image=False)
        return _[0]


