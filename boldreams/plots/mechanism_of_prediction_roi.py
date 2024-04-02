import torch

from utils import stims
import numpy as np
import torchvision
import numpy as np
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
from nibabel.freesurfer.io import *
from utils import closestDivisors,pass_images
from attribution import integrated_gradient_terms
from models.fmri import prep_encoder,channel_stacker
from attribution import integrated_gradient,integrated_gradient_vox

#for V1 we will proceed in the following maner: we have 4 models to consider, AlexNet type I and II and the same for
# vgg. Then we will pick one image and one voxel and look at the rf fields and feature dreams

#lets get the alexnet off model
#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
df=dataFetcher(config['base_path'])
#get data
df.fetch(upto=config['UPTO'])
sub='subj01'
freesurfer_path = base + '/nsd/freesurfer/'

#set up the encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)
enc.load_state_dict(torch.load(model_save_path(config)),strict=False)

#training imgs
imgs=p_enc.train_data.dataset.tensors[0]

#choose a roi to play with
ROI='floc-faces'
roi_filter=df.dic['roi_dic_combined'][ROI]

#get the response to all the images
fmri_test = pass_images(imgs, enc, enc.Nv,batch_size=8)

#figure out top images
img_ind=3
fmri_roi=fmri_test[:,roi_filter].mean(-1)
roi_inds=fmri_roi.argsort(descending=True)
plt.subplots()[1].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))

#figure out top voxels (for V1 voxels make more sense)
roi_vox_inds=fmri_test[roi_inds[img_ind],roi_filter].argsort(descending=True)

#integrated gradients for this voxel
#this will be for all features
#itegrated gradient with terms
voxs=[500]#[13,22,26]
for vox in voxs:
        #@vox=i
        f=0
        fmri_=enc(imgs[img_ind].cuda().float().unsqueeze(0))
        fmri_vox=fmri_[:,df.dic['roi_dic_combined'][ROI]][:,roi_vox_inds[vox]]

        ig=integrated_gradient_vox(enc)

        test=ig(imgs[roi_inds[img_ind]].cuda().float().unsqueeze(0),
                df.dic['roi_dic_combined'][ROI],
                vox=roi_vox_inds[vox])
        fig,ax=plt.subplots(2,3)
        ax[0,0].imshow(test.moveaxis(0,-1).mean(-1))
        ax[0,1].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))
        ax[0,2].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))
        ax[0,2].imshow(test.moveaxis(0,-1).mean(-1),alpha=0.8)
        ax[1,0].imshow(enc.rfs[0][:,:,roi_filter][:,:,roi_vox_inds[vox]].detach().cpu())
        ax[1,1].imshow(enc.rfs[1][:,:,roi_filter][:,:,roi_vox_inds[vox]].detach().cpu())
        ax[1,2].imshow(enc.rfs[2][:,:,roi_filter][:,:,roi_vox_inds[vox]].detach().cpu())
[x.set_axis_off() for x in ax.flatten()]
#make a plot for the figure, we will show how V1 makes  a prediction
#make the mask atop the image
fig,ax=plt.subplots(1,2)
ax[0].set_axis_off()
ax[1].set_axis_off()
mask=test
mask = mask.abs().mean(0)
mask = np.clip(2.8 * mask / mask.max(), 0, 1)
#mask[mask < 0.5] = np.nan
mask = GaussianBlur(21, sigma=(4.5, 4.5))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
ax[0].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))
ax[1].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))
ax[1].imshow(mask,cmap='gray',alpha=0.80)

#okay so lets summarize
 #this is the voxel, roi_vox_inds[vox] gives the voxel
#ROI is already defined and we have filter as roi filer
#to get image use roi_inds[img_ind] is the way to get image index

def mask_p(mask):
        mask = mask.abs().mean(0)
        mask = np.clip(2.8 * mask / mask.max(), 0, 1)
        # mask[mask < 0.5] = np.nan
        mask = GaussianBlur(21, sigma=(4.5, 4.5))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return mask

def row_(model_config,img_id,roi_filter,ax,ax_ind):
    # set up the encoder
    p_enc = prep_encoder(model_config, df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = p_enc.get_encoder(device)
    print("loading:",model_save_path(model_config))
    enc.load_state_dict(torch.load(model_save_path(model_config)), strict=False)

    imgs = p_enc.train_data.dataset.tensors[0]
    img=imgs[img_id].cuda().float().unsqueeze(0)

    print('predicted:',enc(img)[:,roi_filter].mean())
    print( 'true:',p_enc.train_data.dataset.tensors[1][img_id,roi_filter].mean())

    #get the mask
    ig = integrated_gradient(enc)

    mask1 = mask_p(ig(img,
              roi_filter))

    enc_terms=p_enc.get_encoder_terms(device)
    enc_terms.load_state_dict(torch.load(model_save_path(model_config)),strict=False)

    terms_test=enc_terms(img)
    V1=terms_test[:,:,roi_filter][:,:].mean(-1)
    terms_inds=V1.argsort(descending=True)[0]

    layer_stack, channel_stack = channel_stacker(enc.channels_2d)

    dreams=[]
    for i in range(0,4):
        thresholds = 545
        jitter_only = [RandomAffine(5, translate=[0.13, 0.13])]  # ,scale=[1,1], fill=0.0)]
        layer, channel = layer_stack[terms_inds[i]], channel_stack[terms_inds[i]]
        _ = render.render_vis(enc.model, d2u([str(layer)])[0] + ':' + str(channel), show_image=False,
                              thresholds=(thresholds,),
                              transforms=jitter_only)

        dreams.append(_[0][0])

    ax[ax_ind,0].imshow(img[0].moveaxis(0,-1).detach().cpu())
    ax[ax_ind,0].imshow(mask1,alpha=0.8,cmap='gray')
    ax[ax_ind,0].set_axis_off()
    for d,dream in enumerate(dreams):
        ax[ax_ind, d+1].imshow(dream)
        ax[ax_ind, d + 1].set_axis_off()
    bars=V1[:,terms_inds[:25]][0].detach().cpu().numpy()
    ax[ax_ind,-1].bar(np.arange(0,25),bars/bars.max())

fig,ax=plt.subplots(6,6)

# row_(load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json'),
#      #roi_vox_inds[vox],
#      roi_inds[img_ind],
#      roi_filter,
#      ax,0)
#
# row_(load_config(base+'/configs/alexnet_medium_bbtrain-on_max-channels-100.json'),
#      #roi_vox_inds[vox],
#      roi_inds[img_ind],
#      roi_filter,
#      ax,1)
#
#
# #fig,ax=plt.subplots(2,6)
#
# row_(load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-False_max-channels_100.json'),
#      #roi_vox_inds[vox],
#      roi_inds[img_ind],
#      roi_filter,
#      ax,2)
#
# row_(load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-True_max-channels_100.json'),
#      #roi_vox_inds[vox],
#      roi_inds[img_ind],
#      roi_filter,
#      ax,3)


row_(load_config(base+'/configs/bb-RN50_clip_add_last_upto-16_bbtrain-False_max-channels_100.json'),
     #roi_vox_inds[vox],
     roi_inds[img_ind],
     roi_filter,
     ax,4)

row_(load_config(base+'/configs/bb-RN50_clip_add_last_upto-16_bbtrain-True_max-channels_100.json'),
     #roi_vox_inds[vox],
     roi_inds[img_ind],
     roi_filter,
     ax,5)

p_enc = prep_encoder(load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-True_max-channels_100.json'), df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = p_enc.get_encoder(device)
enc.load_state_dict(torch.load(model_save_path(load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-True_max-channels_100.json'))), strict=False)

# #itegrated gradient with terms
# terms_test=enc_terms(imgs[img_ind].cuda().float().unsqueeze(0))
# V1=terms_test[:,:,df.dic['roi_dic_combined'][ROI]][:,:,roi_vox_inds[vox]]
# terms_inds=V1.argsort(descending=True)[0]

# ig=integrated_gradient_terms(enc_terms)
#
# test=ig(imgs[roi_inds[img_ind]].cuda().float().unsqueeze(0),
#         terms_inds[f],
#         df.dic['roi_dic_combined'][ROI],
#         vox=roi_vox_inds[vox])
# fig,ax=plt.subplots(2,3)
# ax[0,0].imshow(test.moveaxis(0,-1).mean(-1))
# ax[0,1].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))
# ax[0,2].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))
# ax[0,2].imshow(test.moveaxis(0,-1).mean(-1),alpha=0.8)
# ax[1,0].imshow(enc.rfs[0][:,:,roi_filter][:,:,roi_vox_inds[vox]].detach().cpu())
# ax[1,1].imshow(enc.rfs[1][:,:,roi_filter][:,:,roi_vox_inds[vox]].detach().cpu())
# ax[1,2].imshow(enc.rfs[2][:,:,roi_filter][:,:,roi_vox_inds[vox]].detach().cpu())


#okay we have the masks, now lets get the dreams of the features
#channel and layer stack

#ok dream cnn feauters now
# i=4
#
# plt.figure()
# plt.imshow(_[0][0])
#
# #plot the terms decay
# plt.figure()
# plt.plot( V1[:,terms_inds].detach().cpu().numpy()[0])