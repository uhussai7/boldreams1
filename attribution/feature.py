import numpy as np

from boldreams import dream_wrapper,dream_wrapper_terms
from utils.roi_utils import remove_keys,greater_than_zero
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.fmri import encoder_terms, channel_stacker


def max_activation_stim_fmri(enc,rois,imgs,batch_size=2,device='cpu'):
    #device=enc.device
    #brain_rois=remove_keys(df.brain_rois(),'other')
    #brain_rois.update(greater_than_zero(remove_keys(df.manual_localizer_rois())))
    if isinstance(enc,encoder_terms):
        n_c=channel_stacker(enc.channels_2d)[0].__len__()
        dreamer=dream_wrapper_terms(enc,rois)
        outs={key:torch.zeros([len(imgs),n_c,values.sum()]) for (key,values) in rois.items()}
    else:
        dreamer=dream_wrapper(enc,rois)
        outs={key:torch.zeros([len(imgs),values.sum()]) for (key,values) in rois.items()}
    for i in tqdm(range(0,len(imgs),batch_size)):
        out=dreamer(imgs[i:i+batch_size].to(device))
        for key in outs.keys():
            #for j in range(0,batch_size):
            outs[key][i:i+batch_size]=out[key][:].detach().cpu()
    return outs

def max_activation_stim_fmri_gt(roi_dic,fmris):
    outs={}
    for key,val in roi_dic.items():
        outs[key]=torch.from_numpy(fmris[:,val])
    return outs

def max_activation_stim_inds(fmri):
    inds={}
    for key,value in fmri.items():
        inds[key]=fmri[key].mean(-1).argsort(descending=True)
    return inds

def plot_max_images(inds,images,N=4):
    n=int(np.sqrt(N))
    fig,ax=plt.subplots(n,n)
    i=0
    for a in range(0,n):
        for b in range(0,n):
            ax[a,b].imshow(images[int(inds[i])].moveaxis(0,-1).detach().cpu().numpy())
            i+=1











