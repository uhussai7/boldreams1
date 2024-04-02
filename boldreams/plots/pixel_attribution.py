import numpy as np
import sys
import torch
from utils import *
from attribution import max_activation_stim_fmri,integrated_gradient,max_activation_stim_inds,plot_max_images,integrated_gradient_channel
from models.fmri import prep_encoder,channel_stacker
from boldreams import dream_wrapper, dream_wrapper_terms
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from boldreams.objectives import *
from torchvision.utils import save_image
from dataHandling import dataFetcher
import matplotlib

#NOTE: things in feature_attribution.py go well with things here.

#paths
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
df=dataFetcher(config['base_path'])

#jsut get one session ofr images
UPTO_=16 #warning: changing upto for speed, sample should be large enough
df.fetch(upto=UPTO_)

#rois
rois=df.dic['roi_dic_combined']

#get the correct encoder to get the ordering
p_enc = prep_encoder(config, df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = load_checkpoint(config,p_enc.get_encoder(device).float().cuda())

#rois
rois=df.dic['roi_dic_combined']

#
r='V2'
ii=0
#stimulus ordering
fmrip_all=max_activation_stim_fmri(enc,rois,p_enc.train_data.dataset.tensors[0],device='cuda')
inds=max_activation_stim_inds(fmrip_all)
#plot some max images
plot_max_images(inds[r],p_enc.train_data.dataset.tensors[0],N=16)
#save some images
for i in range(0,10):
    img=p_enc.train_data.dataset.tensors[0][inds[r][ii]]
    save_image(img,'/home/uzair/Documents/%s_%s_%d.png'%(str(config['backbone_name']),str(r),i))


#integrated gradient stuff
ii=2
r='V2'
masks=[]
dreamer=dream_wrapper(enc,rois)
img=p_enc.train_data.dataset.tensors[0][inds[r][ii]].unsqueeze(0)
ig=integrated_gradient(dreamer)
mask=ig(img.cuda(),r).detach().cpu()
mask=mask.abs().mean(0)
mask=GaussianBlur(21,sigma=(2.5,2.5))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
mask=np.clip(2.1*mask/mask.max(),0,1)
mask[mask<0.4]=np.nan

plt.figure()
plt.imshow(p_enc.train_data.dataset.tensors[0][inds[r][ii]].moveaxis(0,-1).detach().cpu())
plt.imshow(mask,alpha=0.5)




#change encoder to to terms
p_enc = prep_encoder(config, df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = load_checkpoint(config,p_enc.get_encoder_terms(device).float().cuda())

#plot the spikes
import matplotlib
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 22}
r='V1'
ii=6
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


#dreamer=dream_wrapper_terms(enc,rois)
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}
jitter_only = [RandomAffine(4, translate=[0.01, 0.01])]  # ,scale=[1,1], fill=0.0)]

matplotlib.rc('font', **font)
masks=[]
fig, ax = plt.subplots(2,6)
ax[0,0].set_axis_off()
ax[1,0].set_axis_off()
img=p_enc.train_data.dataset.tensors[0][inds[r][ii]].unsqueeze(0)
ax[0,0].imshow(img.squeeze(0).moveaxis(0, -1))
colors=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
thresholds=516
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
    ax[0,i+1].set_axis_off()
    out_img=img.squeeze(0).moveaxis(0, -1)
    #out_img[np.isnan(mask)==1,:]=np.nan
    ax[0,i+1].imshow(out_img)
    ax[0,i+1].imshow(mask,alpha=0.5,cmap='jet')#cmap='jet')
    ax[0,i+1].set_title(str(layer)+'_'+str(channel))
    masks.append(mask)
    ax[1, i + 1].set_axis_off()
    _ = render.render_vis(enc.model, d2u([str(layer)])[0] + ':' + str(channel), show_image=False, thresholds=(thresholds,),
                      transforms=jitter_only)
    #aax=ax[1,i+1].imshow(_[0][0])
    imshow_border(ax[1,i+1],_[0][0],lw=4)

plt.figure()
plt.imshow(img.squeeze(0).moveaxis(0,-1))
colors=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
for i,mask in enumerate(masks):
    plt.imshow(mask,alpha=0.6,cmap=colors[i])


#how can we alse visulize these features?
i=4
layer,channel=layer_stack[spikes__inds[i]],channel_stack[spikes__inds[i]]


plt.figure()
plt.imshow(_[0][0])