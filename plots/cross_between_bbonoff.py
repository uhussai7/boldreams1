import numpy as np
import torch
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

#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-on_max-channels-100.json')
#config=load_config(base+'/configs/bb-vgg11_upto-16_bbtrain-False_max-channels_100.json')

df=dataFetcher(config['base_path'])
#get data
df.fetch(upto=config['UPTO'])
sub='subj01'
freesurfer_path = base + '/nsd/freesurfer/'

#set up the encoder

p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)
#config['off2on']='True'#
enc.load_state_dict(torch.load(model_save_path(config)),strict=False)


i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
face_rois=roi_from_nii(base+'/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/floc-faces.nii.gz',i,j,k,
             freesurfer_path + sub + '/label/floc-faces.mgz.ctab')


#set up the dreaming
#rois=face_rois#df.dic['roi_dic_combined']
rois=df.dic['roi_dic_combined']

#load the image
img=torch.load(base+'/dreams/alexnet-bb-train-on_roi-faces.png').moveaxis(-1,0).unsqueeze(0)
fmri=enc(img.cuda()).detach().cpu()[:,rois['floc-faces']].mean()
print(fmri)

img=torch.load(base+'/dreams/alexnet-bb-train-off_roi-faces.png').moveaxis(-1,0).unsqueeze(0)
fmri=enc(img.cuda()).detach().cpu()[:,rois['floc-faces']].mean()
print(fmri)

#load validation images
# make predictions on validation data
batch_size = 1
val_data = df.validation_data_loader(image_size=p_enc.input_size, batch_size=batch_size)
fmri_pred = torch.zeros_like(val_data.dataset.tensors[1])
fmri_gt = torch.zeros_like(val_data.dataset.tensors[1])
for i, (img, _) in tqdm(enumerate(val_data), total=len(val_data)):
    # print(i*batch_size,(i+1)*batch_size)
    fmri_gt[i * batch_size:(i + 1) * batch_size] = _
    fmri_pred[i * batch_size:(i + 1) * batch_size] = enc(img.cuda()).detach().cpu()

torch.save(fmri_pred,base+'/corrs/alexnet_fmri_pred_bb-on')


#so lets compare the corelation in FFA
fmri_bbon=torch.load(base+'/corrs/alexnet_fmri_pred_bb-on')
fmri_bboff=torch.load(base+'/corrs/alexnet_fmri_pred_bb-off')

#get the roi
fmri_on=fmri_bbon[:,rois['floc-faces']]
fmri_off=fmri_bboff[:,rois['floc-faces']]
fmri=fmri_gt[:,rois['floc-faces']]

corr_on=torch.zeros(len(fmri))
corr_off=torch.zeros(len(fmri))
for pp in range(0,len(fmri)):
    corr_on[pp]=torch.corrcoef(torch.stack([fmri_on[pp,:],fmri[pp,:]]))[0,1]
    corr_off[pp]=torch.corrcoef(torch.stack([fmri_off[pp,:],fmri[pp,:]]))[0,1]

plt.figure()
plt.scatter(corr_on,corr_off)
plt.plot([-0.1,0.4],[-0.1,0.4],color='r')