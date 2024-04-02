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

#lets get an encoding model
#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
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

#lets visulize the rf fields for V1 voxels
V1filter=df.dic['roi_dic_combined']['V1']
noise=df.dic['voxel_ncsnr'][V1filter]
noise_filter=noise>0.8
rfsV1=[rf[:,:,V1filter][:,:,noise_filter].detach().cpu() for rf in enc.rfs]

#lets extract rfs in in x-y windows
NH=10
H=rfsV1[0].shape[0]
dH=H/(NH-1)
fig,ax=plt.subplots(NH,NH)
inds=torch.randint(0,rfsV1[0].shape[-1],(int(NH*NH),))
k=0
for i in range(0,NH):
    for j in range(0,NH):
        # x_start=i*dH
        # y_start=j*dH
        # for k in range(0,rfsV1[0].shape[-1]):
        #     img=rfsV1[0][:,:,k].abs()
        #     x0_,y0_=torch.where(img>0.1)
        #     for x0,y0 in zip(x0_,y0_):
        #         if (x0-x_start).abs()<dH/4 and (y0-y_start).abs()<dH/4:
        #             if img.std()<0.017:
        #                 ax[i,j].imshow(img.abs(),'gray')
        ax[i, j].set_axis_off()
        ax[i, j].imshow(rfsV1[0][:,:,inds[k]].abs(), 'gray')
        k+=1

fig,ax=plt.subplots(10,10)
k=0
for i in range(0,10):
    for j in range(0,10):
        ax[i,j].imshow(rfsV1[0][:,:,100+k],cmap='gray')
        k+=1

#we have to make some rings
ring_width=30

to_tensor=torchvision.transforms.PILToTensor()
ecc_stims=[]
for i in range(0,4):
    s = stims.retino(img_size=p_enc.input_size[-2:])
    s.checker_wedge(ring_width*i,ring_width*(i+1), 0, 360, 4, 20)
    g=to_tensor(s.img)/255
    ecc_stims.append(g)

fig,ax=plt.subplots(1,len(ecc_stims))
for p in range(0,len(ecc_stims)):
    ax[p].set_axis_off()
    ax[p].imshow(ecc_stims[p].moveaxis(0,-1))

plt.imshow(ecc_stims[0].moveaxis(0,-1))

ecc_stims=torch.stack(ecc_stims,dim=0)

fmri_ecc_stims=enc(ecc_stims.float().cuda()).detach().cpu()-enc(0.5*torch.ones_like(ecc_stims).float().cuda()).detach().cpu()

#curv_info=read_label(freesurfer_path + sub + '/label/')

#flat map'
#get surfs first we have to normalize them somehow to show on one plot
ecc_surfs=[]
i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
for p in range(0,len(ecc_stims)):
    ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
    fmri_ecc_nii=array_to_func1pt8(i,j,k,fmri_ecc_stims[p],ref_nii)
    fmri_ecc_surf=func1pt8_to_surfaces(fmri_ecc_nii,base+'/nsd/',sub,method='linear')[0]
    ecc_surfs.append(fmri_ecc_surf)

#lets normalize them between [1,len(ecc_surfs] and set a cut-off of 1std
flat_hemis=get_flats(freesurfer_path,sub)
ecc_final_surf=[np.zeros_like(a) for a in fmri_ecc_surf]
ecc_ind_surf=[]
mask=[]
for p in range(0,len(ecc_surfs)):
    ecc_surf=[]
    this_mask=[]
    for hemi in range(0,2):
        this_surf=( ecc_surfs[p][hemi])/ecc_surfs[p][hemi].max()
        this_surf[this_surf > (this_surf.mean()+ 1.5*this_surf.std())] = 1
        this_surf[this_surf < 1] = 0#np.NaN
        #this_surf=this_surf/this_surf.max()
        this_surf=(2*p+3)*this_surf
        print(2*p+3)
        ecc_surf.append(this_surf)
        ecc_final_surf[hemi][this_surf>0]=0
        ecc_final_surf[hemi]=ecc_final_surf[hemi] + this_surf
        this_mask.append(this_surf>0)
    ecc_ind_surf.append(ecc_surf)
    mask.append(this_mask)
    make_flat_map(ecc_surf, flat_hemis, base + '/dream_figures/retino_ecc%d.png' % p, colormap='jet')
#ecc_final_surf[0][ecc_final_surf[0]>5]=0
#ecc_final_surf[1][ecc_final_surf[1]>5]=0
make_flat_map(ecc_final_surf,flat_hemis,base+'/dream_figures/retino_ecc%d.png'%p,colormap='jet',vmin=0,vmax=10.6)

#lets make it with a color map
min_value=3
max_value=7
colormap = plt.cm.viridis  # You can choose a different colormap
norm = plt.Normalize(vmin=min_value, vmax=max_value)
mlab.figure()
for p in range(0,len(ecc_surfs)):
    for hemi in range(0,2):
        colors=colormap(ecc_ind_surf[p][hemi])
        colors_=tuple([tuple(c[:3]) for c in colors])
        mlab.triangular_mesh(flat_hemis[hemi][0][:,0],flat_hemis[hemi][0][:,1],flat_hemis[hemi][0][:,2],
                             flat_hemis[hemi][1],scalars=ecc_ind_surf[p][hemi],
                             mask=mask[p][hemi])


#lets look at he manually drawn eccrois
flat_hemis=get_flats(freesurfer_path,sub)
ecc_manual=nib.load(base+'/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/prf-eccrois.nii.gz')
ecc_manual_data=np.zeros_like(ecc_manual.get_fdata())
for p in range(1,5):
    ecc_manual_data[ecc_manual.get_fdata()==p]=(2*(p-1)+3)
ecc_manual_nii=nib.Nifti1Image(ecc_manual_data,ecc_manual.affine)
ecc_manual_surf = func1pt8_to_surfaces(ecc_manual_nii, base + '/nsd/', sub, method='nearest')[0]
make_flat_map(ecc_manual_surf,flat_hemis,base+'/dream_figures/retino_ecc_manual.png',colormap='jet',vmin=0,vmax=10.6)

#now lets dream these rois
#set up the dreaming
optimizer=lambda params: Adam(params,lr=1e-3)
param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01)
i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
fig,ax=plt.subplots(1,4)
for p in range(1,5):
    filter = ecc_manual.get_fdata()[i,j,k]==p
    rois={'ecc':filter}
    dreamer=dream_wrapper(enc,rois)
    jitter_only= [RandomAffine(5,translate=[0.01,0.01])]#,scale=[1,1], fill=0.0)]
    obj = roi("roi_ecc")  #- 1e-5* diversity("roi_ffa") #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([
    # -2]).cuda())
    ##rendering and plotting
    _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
                optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(1426,),show_image=False)
    #[plt.subplots()[1].imshow(_[-1][0]) for i in range(0,1)]
    ax[p-1].set_axis_off()
    ax[p-1].imshow(_[-1][0])