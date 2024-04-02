#clip with back bone training on is weird lets investigate itimport torch

#from utils import stims
import numpy as np
import torch
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


#get config
base='/home/uzair/nvme/'
#config=load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
config=load_config(base+'/configs/bb-RN50_clip_add_last_upto-16_bbtrain-True_max-channels_100.json')
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
val_data = df.validation_data_loader(image_size=p_enc.input_size)
imgs = val_data.dataset.tensors[0]
fmri_gt=val_data.dataset.tensors[1]

fmri_model=torch.zeros_like(fmri_gt)

for i,img in tqdm(enumerate(imgs)):
    fmri_model[i]=enc(imgs[i].unsqueeze(0).cuda().float()).detach().cpu()

nv=fmri_model.shape[-1]
corrt=torch.zeros(nv)
for i in range(0,nv):
    corrt[i]=torch.corrcoef(torch.stack([fmri_model[:,i],fmri_gt[:,i]]))[0,1]

plt.figure()
plt.hist(corrt.detach().cpu().flatten().numpy())

# #choose a roi to play with
# ROI='floc-faces'
# roi_filter=df.dic['roi_dic_combined'][ROI]
#
#
# model=enc.model
#
# enc=''
#
# from lucent.modelzoo.util import get_model_layers
#
# thresholds = 745
# optimizer=lambda params: SGD(params,lr=3)
# #optimizer=lambda params: Adam(params,lr=1e-2)
# param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=1)
# jitter_only = [RandomAffine(5, translate=[0.13, 0.13] ,scale=[0.6,1.3], fill=0.0)]
# _ = render.render_vis(model,'layer4_0_relu3:100', show_image=False,
#                       thresholds=(thresholds,),
#                       transforms=jitter_only,param_f=param_f,optimizer=optimizer,
#                       fixed_image_size=p_enc.input_size[-1])
#
# plt.figure()
# plt.imshow(_[0][0])
#
# #get the response to all the images
# fmri_test = pass_images(imgs, enc, enc.Nv,batch_size=1)
#
# #figure out top images
# img_ind=3
# fmri_roi=fmri_test[:,roi_filter].mean(-1)
# roi_inds=fmri_roi.argsort(descending=True)
# plt.subplots()[1].imshow(imgs[roi_inds[img_ind]].moveaxis(0,-1))