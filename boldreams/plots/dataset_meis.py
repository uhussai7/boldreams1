import numpy as np
import torchvision.io.image

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
from torchvision.io import write_png
from PIL import Image
import torchvision.transforms as transforms


#backbones
backbones={'alexnet_on':'alexnet_medium_bbtrain-on_max-channels-100.json',
           'alexnet_off': 'alexnet_medium_bbtrain-off_max-channels-100.json',
           'vgg_on': 'bb-vgg11_upto-16_bbtrain-True_max-channels_100.json',
            'vgg_off':'bb-vgg11_upto-16_bbtrain-False_max-channels_100.json'}

BB='alexnet_off'
ROI='MT'

#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/'+backbones[BB])

#get data
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])
sub='subj01'
freesurfer_path = base + '/nsd/freesurfer/'

#get the ROI array
print(ROI)
roi_filter=df.dic['roi_dic_combined'][ROI]
print('Using voxels in ROI:%s'%ROI)

#set up the encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)
#enc.load_state_dict(torch.load(model_save_path(config)),strict=False)
enc.load_state_dict(torch.load('/home/uzair/nvme/trainedEncoders/alexnet_check/backbone_name-alexnet_UPTO-16_epochs-10_train_backbone-False_max_filters-False_max_percent-100_trained.pt'),strict=False)

#MEIs
imgs=p_enc.train_data.dataset.tensors[0]
fmri_gt=p_enc.train_data.dataset.tensors[1]
fmri_test = pass_images(imgs, enc, enc.Nv,batch_size=8)
fmri_test_inds=fmri_test[:,roi_filter].mean(-1).argsort(descending=True)
fmri_gt_inds=fmri_gt[:,roi_filter].mean(-1).argsort(descending=True)

#visualize the dataset meis
N=10
fig,axs=plt.subplots(N,N)
for k,ax in zip(fmri_test_inds,axs.flatten()):
    ax.imshow(imgs[k].detach().cpu().moveaxis(0,-1))
    ax.set_axis_off()

N=10
fig,axs=plt.subplots(N,N)
for k,ax in zip(fmri_gt_inds,axs.flatten()):
    ax.imshow(imgs[k].detach().cpu().moveaxis(0,-1))
    ax.set_axis_off()