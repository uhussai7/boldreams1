import numpy as np
import matplotlib.pyplot as plt
import torch
from attribution import filter_attribution
from dataHandling import dataFetcher
from utils.io_utils import load_config,model_save_path
from models.fmri import channel_stacker,prep_encoder
from utils import closestDivisors,pass_images
from matplotlib.colors import Normalize
from matplotlib import patches,cm
from lucent.optvis import render
from torchvision.transforms import RandomAffine


#paths
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-on_max-channels-100.json')
df=dataFetcher(config['base_path'])

#jsut get one session ofr images
UPTO_=1#warning: changing upto for speed, sample should be large enough
df.fetch(upto=UPTO_)

#rois
rois=df.dic['roi_dic_combined']

#get the correct encoder to get the ordering
p_enc = prep_encoder(config, df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)
enc.load_state_dict(torch.load(model_save_path(config)),strict=False)

#get v1 roi
ROI='V1'
roi_filter=df.dic['roi_dic_combined'][ROI]

#get rf fields
v1_rfs=[rf[:,:,roi_filter].detach().cpu() for rf in enc.rfs]
fig,ax=plt.subplots(10,3)
for j in range(0,10):
    [ax[j,i].imshow(v1_rfs[i][:,:,1+5*j]) for i in range(0,3)]

#get the weights
v1_w=enc.w_2d.detach().cpu()[:,roi_filter]
plt.subplots()[1].plot(v1_w[:,115])