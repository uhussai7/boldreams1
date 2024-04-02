from torch.nn.modules.module import Module
import torch
from torch.optim import Adam,SGD
from torchvision.transforms import Compose,RandomAffine
from tqdm import tqdm
from lucent.optvis.param.color import _linear_decorrelate_color

#loss functions
def fmri_target(fmri_target): #target fmri vector
    def inner(fmri_in):
        out=fmri_in-fmri_target
        out=out*out
        return out.mean()
    return inner

def roi_ref_img(roi_filter,sign=-1,gamma=1):
    def inner(fmri,img,ref_img):
        loss1=sign*fmri[:,roi_filter].mean()
        loss2=img-ref_img
        loss2=loss2*loss2
        return loss1+gamma*loss2.mean()
    return inner

def roi_ref_img_target(roi_filter,target,gamma=1):
    def inner(fmri,img,ref_img):
        loss1=fmri[:,roi_filter]-target
        loss1=loss1*loss1
        loss2=img-ref_img
        loss2=loss2*loss2
        return loss1.mean()+gamma*loss2.mean()
    return inner

def activate_roi(roi_fiter): #unbounded mean roi activation
    def inner(fmri_in):
        return -fmri_in[:,roi_fiter].mean()
    return inner

def activate_roi_target(roi_fiter,target_mean): # mean roi activation to target_mean
    def inner(fmri_in):
        loss= target_mean-fmri_in[:,roi_fiter].mean()
        loss=loss*loss
        return loss.mean()
    return inner

def activate_roi_rest_zero(roi_filter,gamma=1): #mean roi activation rest zero
    def inner(fmri_in):
        fmri_roi_loss=-fmri_in[:,roi_filter].mean()
        fmri_not_roi_loss=fmri_in[:,~roi_filter]
        fmri_not_roi_loss=(fmri_not_roi_loss*fmri_not_roi_loss).mean()
        return fmri_roi_loss + gamma*fmri_not_roi_loss
    return inner

def activate_roi_target_rest_zero(roi_filter,target_mean,gamma=1):
    def inner(fmri_in):
        fmri_roi_loss=target_mean-fmri_in[:,roi_filter].mean()
        fmri_roi_loss=(fmri_roi_loss*fmri_roi_loss).mean()
        fmri_not_roi_loss=fmri_in[:,~roi_filter]
        fmri_not_roi_loss=(fmri_not_roi_loss*fmri_not_roi_loss).mean()
        return fmri_roi_loss + gamma*fmri_not_roi_loss
    return inner

class dreamer:
    """
    Module to dream without lucent, this for starting from an image and not noise
    """
    def __init__(self,encoder,lr=2e-2,transforms=None):
        """
        Initializer
        :param encoder: Trained encoder
        :param roi: The roi filter of size Nv
        """
        self.encoder=encoder
        self.transforms=transforms
        self.lr=lr
        if self.transforms is None:
            self.transforms=RandomAffine(4, translate=[0.05, 0.05],scale=(0.4,0.9), fill=0.0)

    def dream(self,img_in,loss_function,optimizer,iterations=1024):
        """
        Dream
        :param img: starting img
        :param iterations: number of iterations for optimization
        :return:
        """
        img=torch.clone(img_in)
        img.requires_grad=True
        optim=optimizer(lr=self.lr,params=[img])
        for i in tqdm(range(0,iterations)):
            #out=self.transforms(torch.vstack([img,img_in]))
            #img,img_in=out[0].unsqueeze(0),out[1].unsqueeze(0)
            fmri=self.encoder(self.transforms(img))
            #img=torch.sigmoid(img)
            loss=loss_function(fmri,img,img_in)
            loss.backward()
            optim.step()
        return torch.clamp(img,0,1).detach().cpu()


