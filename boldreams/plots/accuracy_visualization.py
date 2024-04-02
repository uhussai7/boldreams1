from dataHandling import dataFetcher
from utils.io_utils import get_base_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


#get the the noise
base_path=get_base_path('local')
df=dataFetcher(base_path)
df.fetch(upto=1)
noise=df.dic['voxel_ncsnr']

corr={}
#get the corrs
ps=[1]#,5,10,15,20,25,50,75,100]
for p in ps:
    base_corr_name='backbone_name-alexnet_UPTO-16_epochs-10_train_backbone-True_max_filters-False_max_percent' \
                   '-%d_trained_corr.npy'%p
    corr[p]=np.load(base_path + '/trainedEncoders/' + base_corr_name)

###try out new flattening class
from utils import flat
flt=flat(base_path,df.dic)
flt.project_to_array_overlay(corr[1],'test.png','V1V2V3_outline.png')

#get config
base='/home/uzair/nvme/'
freesurfer_path = base + '/nsd/freesurfer/'
i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
sub='subj01'
for p in ps:
    ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
    faces_nii=array_to_func1pt8(i,j,k,corr[p],ref_nii)
    faces_surf=func1pt8_to_surfaces(faces_nii,base+'/nsd/',sub,method='linear')[0]
    flat_hemis=get_flats(freesurfer_path,sub)
    make_flat_map(faces_surf,flat_hemis,base+'/dream_figures/accuracy_bbtrain-off_%d.png'%p,vmin=0,vmax=0.6)

#make an image with rois
fig,ax=plt.subplots(3,3)
for i,p in enumerate(ps):
    acc=plt.imread(base+'/dream_figures/accuracy_bbtrain-off_%d.png'%p)
    H,L=acc.shape[:2]
    x_off=350
    y_off=200
    early_outline=plt.imread(base+'/overlays/'+ '/V1V2V3-outline.png')
    #plt.figure()
    aa=ax.flatten()[i]
    aa.set_axis_off()
    aa.set_title('%d %% filters per layer'%p)
    aa.imshow(acc[y_off:,int(L/2)-x_off:int(L/2)+x_off])
    aa.imshow(early_outline[y_off:,int(L/2)-x_off:int(L/2)+x_off])