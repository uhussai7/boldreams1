import torch

#from utils import stims
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
import os
#for V1 we will proceed in the following maner: we have 4 models to consider, AlexNet type I and II and the same for
# vgg. Then we will pick one image and one voxel and look at the rf fields and feature dreams

#lets get the alexnet off model
#get config
base='/home/uzair/nvme/'#'/home/u2hussai/projects_u/data/'#'/home/uzair/nvme/'##
config=load_config(base+'/configs/bb-alexnet_upto-16_bbtrain-False_max-channels_100.json')
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

#get ROI
ROI='V1'#'floc-faces'#sys.argv[1]#
roi_dic=df.dic['roi_dic_combined']
roi_filter=roi_dic[ROI]

c_voxels=central_voxels(roi_dic,enc.rfs[0].detach().cpu())
vox=c_voxels[ROI][0]

#get img ids
img_id=img_by_roi()[ROI]
input_img=p_enc.train_data.dataset.tensors[0][img_id]


def mask_p(mask):
        mask = mask.abs().mean(0)
        mask = np.clip(2.8 * mask / mask.max(), 0, 1)
        # mask[mask < 0.5] = np.nan
        mask = GaussianBlur(21, sigma=(4.5, 4.5))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return mask

def row_(model_config,img_id,roi_filter,ax,ax_ind,ax_start,vox_id=None):
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

    enc_terms = p_enc.get_encoder_terms(device)
    enc_terms.load_state_dict(torch.load(model_save_path(model_config)), strict=False)

    #get the mask
    if vox_id is None:
        ig = integrated_gradient(enc)
        mask1 = mask_p(ig(img,roi_filter))
        terms_test = enc_terms(img)
        V1 = terms_test[:, :, roi_filter][:, :].mean(-1)
        terms_inds = V1.argsort(descending=True)[0]
    else:
        #c_voxels=central_voxels(roi_dic,enc.rfs[0].detach().cpu())
        #vox=c_voxels[ROI][0]
        ig = integrated_gradient_vox(enc)
        mask1 = mask_p(ig(img,roi_filter,vox=vox_id))
        terms_test = enc_terms(img)
        V1 = terms_test[:, :, roi_filter][:, :,vox_id]
        terms_inds = V1.argsort(descending=True)[0]


    layer_stack, channel_stack = channel_stacker(enc.channels_2d)

    #optimizer=lambda params: SGD(params,lr=2.8)

    optimizer=lambda params: Adam(params,lr=2e-2)
    param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.06,batch=1)
    dreams=[]
    for i in range(0,4):
        thresholds = 852
        jitter_only = [RandomAffine(5, translate=[0.12, 0.12],scale=[0.75,1.15], fill=0.0)]
        layer, channel = layer_stack[terms_inds[i]], channel_stack[terms_inds[i]]
        _ = render.render_vis(enc.model, d2u([str(layer)])[0] + ':' + str(channel), show_image=False,
                              thresholds=(thresholds,),
                              transforms=jitter_only,
                              optimizer=optimizer,
                              fixed_image_size=mask1.shape[-1],
                              param_f=param_f)

        dreams.append(_[0][0])

    ax[ax_ind,ax_start].imshow(img[0].moveaxis(0,-1).detach().cpu())
    ax[ax_ind,ax_start].imshow(mask1,alpha=0.8,cmap='gray')
    ax[ax_ind,ax_start].set_axis_off()
    for d,dream in enumerate(dreams):
        ax[ax_ind,ax_start+ d+1].imshow(dream)
        ax[ax_ind,ax_start + d + 1].set_axis_off()
    bars=V1[:,terms_inds[:25]][0].detach().cpu().numpy()
    plt.subplots_adjust(wspace=0.3)
    ax[ax_ind,-1].bar(np.arange(0,25),bars/bars.max())
    ax[ax_ind,-1].set_axis_on()
    ax[ax_ind,-1].tick_params(axis='both', labelsize=12)  



#okay make the plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 22,
})
plt.ioff()

backbone_names=[r'\noindent AlexNet \\ Type I',
                r'\noindent AlexNet \\ Type II',
                r'\noindent VGG \\ Type I',
                r'\noindent VGG \\ Type II',
                #r'\noindent CLIP \\ RN50 \\Type I',
                r'\noindent CLIP \\ RN50x4 \\Type I']

# Create a figure
offset=2
scale=2.3
width=scale*(4+4)
height=scale*5
nrows=5
ncols=6+offset
fig,ax = plt.subplots(nrows,ncols,figsize=(width,height))
[a.set_axis_off() for a in ax.flatten()] #turn of all the axies


#place the input figure
hw_input=0.27 #height width of input image
left, bottom, width, height = 0, 0.5-hw_input/2, hw_input, hw_input
ax1 = fig.add_axes([left, bottom, width, height])
ax1.set_title('Input image')
ax1.set_axis_off()
ax1.imshow(input_img.moveaxis(0,-1))


if ROI in ['V1', 'V2', 'V3']:
    vox_id=vox
else:
    vox_id=None
#vox_id=None#vox_id
row_(load_config(base+'/configs/bb-alexnet_upto-16_bbtrain-False_max-channels_100.json'),
     img_id,
     roi_filter,
     ax,0,offset,vox_id)

# row_(load_config(base+'/configs/bb-alexnet_upto-16_bbtrain-True_max-channels_100.json'),
#      img_id,
#      roi_filter,
#      ax,1,offset,vox_id)
#
# row_(load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-False_max-channels_100.json'),
#      img_id,
#      roi_filter,
#      ax,2,offset,vox_id)
#
# row_(load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-True_max-channels_100.json'),
#      img_id,
#      roi_filter,
#      ax,3,offset,vox_id)
#


# row_(load_config(base+'/configs/bb-RN50x4_clip_relu3_last_upto-16_bbtrain-False_max-channels_100.json'),
#      img_id,
#      roi_filter,
#      ax,4,offset,vox_id)

    

for i in range(0,5):
    ax[i, 2].text(-0.48, 0.5, backbone_names[i],transform=ax[i, 2].transAxes,ha='center',va='center')
    for j in range(2,5+3):
        if j==2 and i==0:
            ax[i,j].text(0.5,1.3,r'\begin{center} Pixel \\ attribution \end{center}',ha='center',va='center',
                         transform=ax[i,j].transAxes)
        if j==ncols-1 and i==0:
            ax[i, j].text(0.5, 1.1, r'\begin{center} Decay \end{center}', horizontalalignment='center',
                          transform=ax[i, j].transAxes)

fig.text((offset+3)*(1)/(ncols),0.9,'Feature visualization',ha='center',va='center')



os.makedirs(base+'/plots/',exist_ok=True)
plt.savefig(base+'/plots/mech_of_pred_%s.png'%ROI,dpi=600)
