import clip
from dataHandling import dataFetcher
import sys
import torch
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from torch.nn import MSELoss
from torch.optim import Adam,SGD
from torchvision.models.feature_extraction import get_graph_node_names
from models.fmri import prep_encoder
from boldreams import dream_wrapper
import matplotlib.pyplot as plt
from boldreams.objectives import *
from boldreams.param import ref_image
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render
from torchvision.utils import save_image
import numpy as np

from utils.surface import *
from mayavi import mlab
import sys
from dataHandling import dataFetcher
import nibabel as nib
from utils.roi_utils import anat_combine_rois,vol_for_flat_maps
from utils.io_utils import load_config
from attribution.feature import *

#get config
config=load_config('./configs/alexnet_medium_bbtrain-off.json')

#get data
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])

#get encoders
p_enc_off=prep_encoder(config,df)
enc_off=p_enc_off.get_encoder('cpu')
enc_on=prep_encoder(load_config(('./configs/alexnet_medium_bbtrain-on.json')),df).get_encoder('cpu')
enc_off.load_state_dict(torch.load(model_save_path(config)),strict=False)
enc_on.load_state_dict(torch.load(model_save_path(load_config(('./configs/alexnet_medium_bbtrain-on.json')))))

#get face stimulus
rois=df.dic['roi_dic_combined']
r='floc-faces'
fmrip_all=max_activation_stim_fmri(enc_off.cuda(),rois,p_enc_off.train_data.dataset.tensors[0],device='cuda')
inds=max_activation_stim_inds(fmrip_all)
plot_max_images(inds[r],p_enc_off.train_data.dataset.tensors[0])

#get feature attribution for max image
#have to reload the terms model
enc_off=p_enc_off.get_encoder_terms('cpu')
enc_on=prep_encoder(load_config(('./configs/alexnet_medium_bbtrain-on.json')),df).get_encoder_terms('cpu')
enc_off.load_state_dict(torch.load(model_save_path(config)),strict=False)
enc_on.load_state_dict(torch.load(model_save_path(load_config(('./configs/alexnet_medium_bbtrain-on.json')))))

#check filter activations
face_img=p_enc_off.train_data.dataset.tensors[0][inds[r][5]]
spikes_off=enc_off(face_img.unsqueeze(0))[:,:,rois[r]].mean(-1).detach().cpu()[0]
spikes_on=enc_on(face_img.unsqueeze(0))[:,:,rois[r]].mean(-1).detach().cpu()[0]
#get layerm,filters of the top5
spikes_on_inds=spikes_on.argsort(descending=True)
spikes_off_inds=spikes_off.argsort(descending=True)
layer_stack,channel_stack=channel_stacker(enc_off.channels_2d)
#print some summary
print('top 5 features in backbone training off')
print(np.asarray(layer_stack)[spikes_off_inds[0:5].numpy()],np.asarray(channel_stack)[spikes_off_inds[0:5]])
print('top 5 features in backbone training oon')
print(np.asarray(layer_stack)[spikes_on_inds[0:5].numpy()],np.asarray(channel_stack)[spikes_on_inds[0:5]])
#visulize the dreams
def get_images(cnn,layer,channel,optim='Adam',lr=0.8,sd=0.5,rot=10,trans=0.07,scale=[1.0,1.0],thresholds=(512,)):
    if optim=='Adam':
        optimizer=lambda params: Adam(params,lr=lr)
    elif optim=='SGD':
        optimizer=lambda params: SGD(params,lr=lr)
    param_f = lambda: param.image(p_enc_off.input_size[-1], fft=True, decorrelate=True,sd=sd)
    objectives=layer.split('.')[0] + '_' +layer.split('.')[1] + ':' + str(channel)
    jitter_only = [RandomAffine(rot, translate=[trans, trans]   ,scale=scale, fill=0.0)]
    print(objectives)
    _=render.render_vis(cnn.cuda().eval(),str(objectives),show_image=False,param_f=param_f,optimizer=optimizer,
                           transforms=jitter_only,thresholds=thresholds)
    return _[0][0]

on_imgs_before=[]
on_imgs_after=[]
start=0
top_n=10
for i in range(start,top_n):
    print(i)
    layer,channel=layer_stack[spikes_on_inds[i]],channel_stack[spikes_on_inds[i]]
    on_imgs_before.append(get_images(enc_off.model,layer,channel,lr=4e-3,sd=0.01,rot=15,trans=0.1,thresholds=(1024,)))
    on_imgs_after.append(get_images(enc_on.model,layer,channel,lr=4e-3,sd=0.01,rot=15,trans=0.1,thresholds=(1024,)))
fig,ax=plt.subplots(2,top_n)
for i in range(0,len(on_imgs_before)):
    ax[0,i].imshow(on_imgs_before[i])
    ax[1, i].imshow(on_imgs_after[i])


