import numpy as np
import torch

from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
import sys
from dataHandling import dataFetcher
from utils.io_utils import load_config
from lucent.optvis import render
from boldreams.objectives import *
from models.fmri import prep_encoder
from boldreams import dream_wrapper
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render
from torch.optim import Adam,SGD
from models.fmri import prep_encoder
import matplotlib.pyplot as plt

#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-on_max-channels-100.json')
#config['UPTO']=4
df=dataFetcher(config['base_path'])
#get data
df.fetch(upto=1)#config['UPTO'])
sub='subj01'
freesurfer_path = base + '/nsd/freesurfer/'

#set up the encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)
enc.load_state_dict(torch.load(model_save_path(config)),strict=False)
#enc.load_state_dict(torch.load('/home/uzair/nvme/trainedEncoders/backbone_name-alexnet_UPTO-4_epochs
# -10_train_backbone-False_max_percent-None_max_filters--1_trained.pt'),strict=False)

#rois
rois=df.dic['roi_dic_combined']

rois_to_test=['V1','V2','V3','V3ab','hV4','VO','LO','PHC','IPS','MT','MST']
#get an image
img=p_enc.train_data.dataset.tensors[0][10].unsqueeze(0)

#okay seems like it is working, lets take it one ROI at a time and see where is there infromation of the image


#lets try the naive approach
fmri_target=enc(img.cuda()).detach().cpu()
criterion=MSELoss()
xfm=Compose([GaussianBlur(5), RandomAffine(0,[0.01,0.01])])


input_img = torch.rand(p_enc.input_size)
input_img.requires_grad = True
optimizer = Adam([input_img], lr=0.01, weight_decay=0)
roit=rois[rois_to_test[0]]
cuma_imgs=[]
for i in range(0,8):
    roit=rois[rois_to_test[0]]
    for k in range(0,i):
        print(rois[rois_to_test[k]])
        roit=roit+rois[rois_to_test[k]]
    input_img = torch.rand(p_enc.input_size)
    input_img.requires_grad = True
    optimizer = Adam([input_img], lr=0.007, weight_decay=0)
    for i in range(0,4000):
        optimizer.zero_grad()
        this_fmri=enc(xfm(input_img.cuda()))
        loss=criterion(fmri_target.cuda()[:,roit].flatten(),this_fmri[:,roit].flatten())
        #loss=fmri_target.cuda()-this_fmri
        #loss=loss*loss
        #loss=loss.mean()
        if i%100==0:
            print(loss)
        loss.backward()
        optimizer.step()
    cuma_imgs.append(input_img.detach().cpu())

fig,ax=plt.subplots(1,1+len(cuma_imgs))
ax[0].imshow(img.detach()[0].moveaxis(0, -1).cpu())
ax[0].set_axis_off()
rois_used='V1'
for i in range(0,len(cuma_imgs)):
    if i>0:
        rois_used=rois_used + '+' + rois_to_test[i]
    ax[i+1].set_axis_off()
    imshow_border(ax[i+1],cuma_imgs[i].detach()[0].moveaxis(0,-1).cpu())
    ax[i+1].set_title(rois_used)
    #ax[i+1].imshow(cuma_imgs[i].detach()[0].moveaxis(0,-1).cpu())


re_imgs=[]
for r in rois_to_test:
    input_img = torch.rand(p_enc.input_size)
    input_img.requires_grad = True
    optimizer = Adam([input_img], lr=0.007, weight_decay=0)
    for i in range(0,4000):
        optimizer.zero_grad()
        this_fmri=enc(xfm(input_img.cuda()))
        loss=criterion(fmri_target.cuda()[:,rois[r]].flatten(),this_fmri[:,rois[r]].flatten())
        #loss=fmri_target.cuda()-this_fmri
        #loss=loss*loss
        #loss=loss.mean()
        if i%100==0:
            print(r,loss)
        loss.backward()
        optimizer.step()
    re_imgs.append(input_img.detach().cpu())


fig,ax=plt.subplots(2,len(rois_to_test))
for i in range(0,len(rois_to_test)):
    ax[0,i].imshow(img.detach()[0].moveaxis(0,-1).cpu())
    ax[1,i].imshow(re_imgs[i].detach()[0].moveaxis(0,-1).cpu())


#set up the dreaming
#rois=face_rois#df.dic['roi_dic_combined']
dreamer=dream_wrapper(enc,rois)


#get roi targets
fmri_roi_targets_=dreamer(img.cuda())
fmri_roi_targets={key:fmri_roi_targets_[key].detach().cpu() for key in fmri_roi_targets_.keys()}
del fmri_roi_targets_

optimizer=lambda params: Adam(params,lr=1e-2)
param_f = lambda: param.image(p_enc.input_size[-1], fft=False, decorrelate=False,sd=0.4)
jitter_only= [RandomAffine(0,translate=[0.0005,0.0005])]#,scale=[1,1], fill=0.0)]
obj = roi_targets({key:fmri_roi_targets[key] for key in ['V1','V2','V3']})  #- 1e-5* diversity("roi_ffa") #+
# 1.2*roi_mean_target([
# 'roi_v1'],
# torch.tensor([
    # -2]).cuda())

##rendering and plotting
_=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
                optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(1026,),show_image=False)
[plt.subplots()[1].imshow(_[-1][0]) for i in range(0,1)]

fmri_dream=dreamer(torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1))

fmri_=enc(torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1))
