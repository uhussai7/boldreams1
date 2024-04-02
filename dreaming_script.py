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


buff=sys.argv[0]
sys.argv=[buff]

sys.argv.append('alexnet')#'RN50x4_clip_relu3_last')#'vgg11')#'alexnet')#'RN50x4_clip_relu3_last')
sys.argv.append(16)
sys.argv.append(10)
sys.argv.append(-1)
sys.argv.append('False')
print(sys.argv)
#configure
config={ 'SYSTEM':'local',
         'backbone_name':sys.argv[1],
         'UPTO':int(sys.argv[2]),
         'epochs':int(sys.argv[3]),
         'max_filters':int(sys.argv[4]),
         'train_backbone': sys.argv[5],
         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
         }
config['base_path']=get_base_path(config['SYSTEM'])
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])


#initialize encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)

#load state_dic
train_path=model_save_path(config)
checkpoint=torch.load(train_path)
enc.load_state_dict(checkpoint,strict=False)

#get max_activating images


#Dreamer
rois=df.dic['roi_dic_combined']
dreamer=dream_wrapper(enc,rois)
##optimzier
optimizer=lambda params: Adam(params,lr=1e-3)
#optimizer=lambda params: SGD(params,lr=1.3)
##initial image
param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01)
#param_f = lambda: ref_image(img)
##transforms
jitter_only= [RandomAffine(1,translate=[0.15,0.15])]#,scale=[1,1], fill=0.0)]
##objective
obj = roi("roi_floc-faces") #- 1e-5* diversity("roi_ffa") #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([-2]).cuda())
##rendering and plotting
_=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
                optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(4124,),show_image=False)
[plt.subplots()[1].imshow(_[-1][0]) for i in range(0,1)]

fmri_dream=dreamer(torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1))

#then get top stimuli for faces



#save_image(torch.from_numpy(_[0][0]).moveaxis(-1,0),'/home/uzair/Documents/%s.png'%str(config['backbone_name']))

#we want to project the dream fmri signal
dream_img=torch.from_numpy(_[0][0]).moveaxis(-1,0).unsqueeze(0)
dream_fmri=enc(dream_img.cuda())
#fmri[:,rois['floc-faces']].mean()
face_ind=1953
img=df.training_data_loader(image_size=p_enc.input_size).dataset.tensors[0][face_ind].unsqueeze(0)
fmri=enc(img.cuda())
fmri_gt=df.training_data_loader(image_size=p_enc.input_size).dataset.tensors[1][face_ind].unsqueeze(0)

#projection stuff
subj='subj01'
freesurfer_path=config['base_path'] + '/nsd/freesurfer/' #+ '/' + subj + '/'
func1pt8toanat=NiftiVolume(config['base_path'] + '/nsd/nsddata/ppdata/' + subj +
                           '/transforms/func1pt8-to-anat0pt8.nii.gz')
roi_list=['floc-bodies','floc-faces','floc-places','floc-words']
test,roi_inds=vol_for_flat_maps(roi_list,config['base_path'],subj)
roi_vol=NiftiVolume(nii=test)
curvature=[nib.freesurfer.io.read_morph_data(freesurfer_path + subj + '/surf/lh.curv'),
           nib.freesurfer.io.read_morph_data(freesurfer_path + subj + '/surf/rh.curv')]

flat=[FlatSurface(0,freesurfer_path,subj,'lh'),FlatSurface(0,freesurfer_path,subj,'rh')]
roi_flat=[fla.project_vol(roi_vol) for fla in flat]
mask_flat=[(roi_fla>0).astype(float) for roi_fla in roi_flat]
offset=[0,350]

dream_fmri_flat=[fla.project_fmri_list(dream_fmri[0].detach().cpu(),df.x,df.y,df.z,xfm_vol=func1pt8toanat) for fla in
                 flat]
fmri_flat=[fla.project_fmri_list(fmri[0].detach().cpu(),df.x,df.y,df.z,xfm_vol=func1pt8toanat) for fla in flat]
fmri_flat_gt=[fla.project_fmri_list(fmri_gt[0].detach().cpu(),df.x,df.y,df.z,xfm_vol=func1pt8toanat) for fla in flat]

outs=[dream_fmri_flat,fmri_flat]
for out in outs:
    for h,d in enumerate(out):
        d[mask_flat[h]==0]=np.nan
    mlab.figure()
    mlab.triangular_mesh(flat[h].fpts[:, 0] + offset[h], flat[h].fpts[:, 1], flat[h].fpts[:, 2], flat[h].fpolys,
                         scalars=curvature[h])#, mask=mask_flat[h],
                         #vmin=-2, vmax=2)
    for h in range(0,2):
        mlab.triangular_mesh(flat[h].fpts[:,0]+offset[h],flat[h].fpts[:,1],flat[h].fpts[:,2],flat[h].fpolys,
                         scalars=out[h]*mask_flat[h], mask=mask_flat[h],
                     vmin=-2.5,vmax=2.5)
    mlab.view(0.0, 0.0, 1200)
    mlab.colorbar()

fmri_flat[np.isnan(fmri_flat)]=0
mlab.figure()
mlab.triangular_mesh(flat.fpts[:,0],flat.fpts[:,1],flat.fpts[:,2],flat.fpolys,scalars=fmri_flat*mask_flat,vmin=-2,vmax=2)

#make a mask of the face regions
roi_list=['floc-faces']
test,roi_inds=vol_for_flat_maps(roi_list,config['base_path'],subj)
face_vol=NiftiVolume(nii=test)
face_flat=[fla.project_vol(roi_vol) for fla in flat]
mlab.figure()
for h in range(0,2):
    face_flat[h][face_flat[h]<26]=0
    mlab.triangular_mesh(flat[h].fpts[:,0]+offset[h],flat[h].fpts[:,1],flat[h].fpts[:,2],flat[h].fpolys,
                         scalars=face_flat[h])
mlab.view(0.0, 0.0, 1200)
mlab.colorbar()
##try filtering the mesh based on the mask
points_=[]
tris_=[]
for h in range(0,2):
    values_indices_inv=np.zeros_like(mask_flat[h])
    values_indices_inv[:]=np.nan
    count=0
    for i,value in enumerate(mask_flat[h]):
        if value:
            values_indices_inv[i]=count
            count+=1
    valid_indices = np.where(mask_flat[h])[0]
    triangles=[]
    for triangle in flat[h].fpolys:
        if (mask_flat[h][triangle]).sum()==3:
            triangles.append(values_indices_inv[triangle])
    triangles=np.asarray(triangles)
    points_.append(flat[h].fpts[mask_flat[h]==1])
    tris_.append(triangles)

outs = [dream_fmri_flat, fmri_flat,fmri_flat_gt]
outs_str=['dream_fmri_flat', 'fmri_flat','fmri_flat_gt']
for o,out in enumerate(outs):
    mlab.figure(bgcolor=(1,1,1),fgcolor=(0.,0.,0.),size=(1000,1000))
    for h in range(0, 2):
        scalar = out[h][mask_flat[h] == 1]
        scalar[np.isnan(scalar) == 1] = 0
        mlab.triangular_mesh(points_[h][:, 0] + offset[h], points_[h][:, 1], points_[h][:, 2],
                             tris_[h],
                             scalars=scalar,
                             vmin=-2.5, vmax=2.5)
    mlab.view(0.0, 00.0, 580)
    mlab.colorbar()
    mlab.savefig('/home/uzair/Documents/%s.png'%outs_str[o])



                     #vmin=-2,vmax=10)
#fmri[:,rois['floc-faces']].mean()



#tight plot
# fig, axs = plt.subplots(1, 4)
# # Remove axes, ticks, and labels
# for i,ax in enumerate(axs):
#     ax.axis('off')
#     ax.imshow(imgs[i])
# # Set the space between subplots
# plt.subplots_adjust(wspace=0.01)

#
# #lets just check the regular way first
# from boldreams.wrappers import dream_wrapper,clip_wrapper
# from boldreams.transforms import dynamicTransform
# from boldreams.objectives import *
# import matplotlib.pyplot as plt
# from torchvision.transforms import GaussianBlur,Compose,RandomAffine
# from torchvision.transforms import InterpolationMode
#
# emotions=['a happy photo','Canada day celebrations','a sad photo']
# #rendering for clip
# #the issue is that although the model is in enc its not the the whole model for text
# #steerability
#
#
# face_rois=df.specify_rois(df.manual_roi_folder,['floc-bodies.nii.gz'])
# face_rois={'ffa': face_rois[key]>0 for key in ['floc-bodies']}
# #
# v1_roi=df.manual_retinotopy_rois()
# v1_rois={'ecc':v1_roi['prf-eccrois']==1}
# dreamer=dream_wrapper(enc.cuda().eval(),face_rois).cuda()
# clip_model=clip.load('RN50x4')
#
#
# b=1
# input_size=p_enc.input_size
# maxx=1200
# interval=maxx
# scale_cut=maxx-100
# r=3*torch.ones(maxx+1)
# t=0.01*torch.ones(maxx+1)
# s=1.0*torch.ones(maxx+1)
# s[:scale_cut]=0.3
# t[scale_cut:]=0.1
# N=int((maxx)/interval)
# thresholds=torch.linspace(0,maxx,N+1).int()
# #optimizer=lambda params: Adam(params,lr=5e-2)
# optimizer=lambda params: SGD(params,lr=8.95)
# param_f = lambda: param.image(input_size[-1], fft=True, decorrelate=True,sd=0.03, batch=b)
# jitter_only = [dynamicTransform(r,t,s)]
# jitter_only= [RandomAffine(15,translate=[0.19,0.19],scale=[0.64,0.96], fill=0.0)]
# #obj = roi("roi_ffa") #- 1e-5* diversity("roi_ffa") #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([-2]).cuda())
# obj=clip_img_features(0.2)
# #text steering
# from boldreams.wrappers import clip_wrapper
#
#
# # for emotion in emotions:
# emotion=emotions[0]
# text_dreamer=clip_wrapper(clip_model[0],dreamer,clip.tokenize([emotion]))
#
# _=render.render_vis(text_dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
#                 optimizer=optimizer,fixed_image_size=input_size[-1],thresholds=thresholds,show_image=False)
# [plt.subplots()[1].imshow(_[-1][0]) for i in range(0,b)]