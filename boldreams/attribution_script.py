import numpy as np

from dataHandling import dataFetcher
import sys
import torch
from utils import *
from attribution import max_activation_stim_fmri,integrated_gradient,max_activation_stim_inds,plot_max_images,integrated_gradient_channel
from models.fmri import prep_encoder,channel_stacker
from boldreams import dream_wrapper, dream_wrapper_terms
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur
from boldreams.objectives import *
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render
from torch.optim import Adam,SGD
from lucent.optvis import render
from boldreams.objectives import *
from torchvision.utils import save_image
from torch import softmax

#loading data
sys.argv=sysargs(sys.argv,'RN50x4_clip_relu3_last',6,10,-1,'False')
#sys.argv=sysargs(sys.argv,'alexnet',6,10,-1,'False')

#configure
config={ 'SYSTEM':'local',
         'backbone_name':sys.argv[1],
         'UPTO':int(sys.argv[2]),
         'epochs':int(sys.argv[3]),
         'max_filters':int(sys.argv[4]),
         'train_backbone': sys.argv[5],
         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
         }
config['base_path']=get_base_path(config['SYSTEM'])
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])

#initialize encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)

#load state_dic
train_path=model_save_path(config)
checkpoint=torch.load(train_path)
enc.load_state_dict(checkpoint,strict=False)

#rois
rois=df.dic['roi_dic_combined']

#
r='floc-faces'
ii=0
#stimulus ordering
fmrip_all=max_activation_stim_fmri(enc,rois,p_enc.train_data.dataset.tensors[0])
inds=max_activation_stim_inds(fmrip_all)
#plot some max images
plot_max_images(inds[r],p_enc.train_data.dataset.tensors[0],N=16)
#save some images
for i in range(0,10):
    img=p_enc.train_data.dataset.tensors[0][inds[r][ii]]
    save_image(img,'/home/uzair/Documents/%s_%s_%d.png'%(str(config['backbone_name']),str(r),i))

#dreamer=dream_wrapper_terms(enc,rois)
masks=[]
dreamer=dream_wrapper(enc,rois)
img=p_enc.train_data.dataset.tensors[0][inds[r][ii]].unsqueeze(0)
ig=integrated_gradient(dreamer)
mask=ig(img.cuda(),r).detach().cpu()
mask=mask.abs().mean(0)
mask=GaussianBlur(21,sigma=(2.5,2.5))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
mask=np.clip(1.8*mask/mask.max(),0,1)
mask[mask<0.6]=np.nan

plt.figure()
plt.imshow(p_enc.train_data.dataset.tensors[0][inds[r][ii]].moveaxis(0,-1).detach().cpu())
plt.imshow(mask,alpha=0.5)

#change encoder to to terms
enc=p_enc.get_encoder_terms(device)
enc.load_state_dict(checkpoint,strict=False)
#plot the spikes
import matplotlib
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
plt.figure()
spikes=enc(p_enc.train_data.dataset.tensors[0][inds[r][ii]].cuda().unsqueeze(0)).detach().cpu()[:,:,rois[r]][0]
bias=enc.b[rois[r]].detach().cpu()
spikes_voxavg=spikes.mean(-1)
bias_voxavg=bias.mean()
signal=spikes.mean(-1).sum(0)
spikes_n,spikes__inds=(spikes_voxavg/spikes_voxavg.max()).sort(descending=True)
plt.bar(np.arange(0,20),spikes_n[0:20])
plt.xlabel("Filter")
plt.ylabel("Contribution to BOLD")
#get location in backbone
layer_stack,channel_stack=channel_stacker(enc.channels_2d)
layer,channel=layer_stack[spikes__inds[0]],channel_stack[spikes__inds[0]]

#get the MEI for the top channel
# optimizer=lambda params: SGD(params,lr=4.9)
# param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.03)
# jitter_only= [RandomAffine(8,translate=[0.1,0.1], scale=[0.5,0.96], fill=0.0)]
# _=render.render_vis(enc.model.cuda().eval(),'layer1_3_relu3:74',param_f=param_f,transforms=jitter_only,
#                 optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(2524,),show_image=False)
# [plt.subplots()[1].imshow(_[-1][0]) for i in range(0,1)]


#dreamer=dream_wrapper_terms(enc,rois)
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
masks=[]
fig, ax = plt.subplots(1,6)
ax[0].set_axis_off()
ax[0].imshow(img.squeeze(0).moveaxis(0, -1))
colors=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
for i in range(0,5):
    layer,channel=layer_stack[spikes__inds[i]],channel_stack[spikes__inds[i]]
    img=p_enc.train_data.dataset.tensors[0][inds[r][ii]].unsqueeze(0)
    # ig=integrated_gradient(dreamer)
    ig=integrated_gradient_channel(enc.model.eval())
    mask=ig(img.cuda(),layer,channel).detach().cpu()
    mask=mask.abs().mean(0)
    mask=GaussianBlur(21,sigma=(4.5,4.5))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    mask=np.clip(1.8*mask/mask.max(),0,1)
    mask[mask<0.5]=np.nan
    #mask=np.clip(2.8*mask,0,1)
    ax[i+1].set_axis_off()
    out_img=img.squeeze(0).moveaxis(0, -1)
    #out_img[np.isnan(mask)==1,:]=np.nan
    ax[i+1].imshow(out_img)
    ax[i+1].imshow(mask,alpha=0.5,cmap='jet')#cmap='jet')
    ax[i+1].set_title(str(layer)+'_'+str(channel))
    masks.append(mask)


plt.figure()
plt.imshow(img.squeeze(0).moveaxis(0,-1))
colors=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
for i,mask in enumerate(masks):
    plt.imshow(mask,alpha=0.6,cmap=colors[i])