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

BB='vgg_on'
ROI='V1'

#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/'+backbones[BB])

#get data
df=dataFetcher(config['base_path'])
df.fetch(upto=1)#config['UPTO'])
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
enc.load_state_dict(torch.load(model_save_path(config)),strict=False)

#set up the dreaming
rois={ROI: roi_filter}
dreamer=dream_wrapper(enc,rois)
#optimizer=lambda params: SGD(params,lr=0.8)
optimizer=lambda params: Adam(params,lr=1e-2)
param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=4)
jitter_only= [RandomAffine(1,translate=[0.06,0.06])]#,scale=[0.9,1.1], fill=0.0)]
obj = roi("roi_"+ROI) - 1e-1* diversity("roi_"+ROI) #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([
# #     # -2]).cuda())
# #obj=roi_mean_target(["roi_"+ROI],torch.tensor([6.0]).cuda())
# # obj=roi_mean_target(["roi_"+'FFA-1'],torch.tensor([0.0]).cuda())+roi_mean_target(["roi_"+'FFA-2'],torch.tensor([
# #     6.0]).cuda()) + roi_mean_target(["roi_"+'OFA'],torch.tensor([0.0]).cuda())
# # obj=1.0*roi_mean_target(["roi_"+ROI],torch.tensor([5.0]).cuda())#+ 1.5*roi_mean_target(["roi_"+'not-faces'],
#                                                                  #                     torch.tensor([
#     #0.0]).cuda())
##dream
_=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
                optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(350,),show_image=False)
for img in _[0]:
    plt.figure()
    plt.imshow(img)


# #get finer rois
# i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
# face_rois=roi_from_nii(base+'/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/floc-faces.nii.gz',i,j,k,
#              freesurfer_path + sub + '/label/floc-faces.mgz.ctab')
#
#
# #set up the dreaming
# rois=df.dic['roi_dic']
# not_faces=np.abs(df.dic['roi_dic_combined'][ROI]-1)
# rois={'floc-faces': torch.ones(np.sum(roi_filter))==1}
# dreamer=dream_wrapper(enc,rois)
# #optimizer=lambda params: SGD(params,lr=0.8)
# optimizer=lambda params: Adam(params,lr=1e-3)
# param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=4)
# jitter_only= [RandomAffine(1,translate=[0.06,0.06])]#,scale=[0.9,1.1], fill=0.0)]
# obj = roi("roi_"+ROI) - 1e-5* diversity("roi_"+ROI) #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([
# #     # -2]).cuda())
# #obj=roi_mean_target(["roi_"+ROI],torch.tensor([6.0]).cuda())
# # obj=roi_mean_target(["roi_"+'FFA-1'],torch.tensor([0.0]).cuda())+roi_mean_target(["roi_"+'FFA-2'],torch.tensor([
# #     6.0]).cuda()) + roi_mean_target(["roi_"+'OFA'],torch.tensor([0.0]).cuda())
# # obj=1.0*roi_mean_target(["roi_"+ROI],torch.tensor([5.0]).cuda())#+ 1.5*roi_mean_target(["roi_"+'not-faces'],
#                                                                  #                     torch.tensor([
#     #0.0]).cuda())
# ##dream
# _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
#                 optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(2350,),show_image=False)
#
# #normalize the image
# fig,ax=plt.subplots(1,len(_[0]))
# for p,img in enumerate(_[0]):
#     #img=torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1)
#     img=img/img.max()
#     if len(_[0])==1:
#         ax.imshow(img)
#     else:
#         ax[p].imshow(img)
#
# ##get fmri and normalize with early signal
# rois=df.dic['roi_dic_combined']
# fmri_=enc(img)[0]
# fmri_=(fmri_-fmri_[np.clip(rois['V1'],0,1)].mean())#/fmri_[np.clip(rois['V1']+rois['V2']+rois[
#     #'V3'],0,1)].std()
#
# #save the dream
# img_pil=transforms.ToPILImage()(torch.from_numpy(_[-1][0]).moveaxis(-1,0))
# img_pil.save(base+'/dreams/dream_roi-%s_bb-%s.png'%(ROI,BB))
# fmri_dream=dreamer(torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1))
#
#
# #project onto flat brain
# ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
# nii=array_to_func1pt8(i,j,k,fmri_.detach().cpu(),ref_nii)
# surf=func1pt8_to_surfaces(nii,base+'/nsd/',sub,method='linear')[0]
# flat_hemis=get_flats(freesurfer_path,sub)
#
# flat_map_name='flat_map_roi-%s_bb-%s.png'%(ROI,BB)
# make_flat_map(surf,flat_hemis,base+'/dream_figures/'+flat_map_name,vmin=-3,vmax=4)
#
# #make an image with rois
# ffa_fmri_dream=plt.imread(base+'/dream_figures/'+flat_map_name)
# ffa_outline=plt.imread(base+'/overlays/'+ '/FFA-outline.png')
#
# plt.figure()
# plt.imshow(ffa_fmri_dream)
# plt.imshow(ffa_outline,alpha=0.8,cmap='Greys')
#
# plt.savefig(base+'/dream_figures/'+flat_map_name)
#
# #lets do bar plot
# rois_for_bar_plot=['V1','V2','V3','floc-faces']
# fmri_bars=[]
# for r in rois_for_bar_plot:
#     fmri_bars.append(fmri_[rois[r]].mean().detach().cpu())
#
# fig,ax=plt.subplots()
# ax.bar(rois_for_bar_plot, fmri_bars)
#ffa_outline=plt.imread(base+'/overlays/'+ '/FFA-1.png') + plt.imread(base+'/overlays/'+ '/FFA-2.png') + plt.imread(
# base+'/overlays/'+ '/OFA.png')
# eba_outline=plt.imread(base+'/overlays/'+ '/FFA-2.png')
# ofa_outline=plt.imread(base+'/overlays/'+ '/OFA.png')



# outline=np.abs(np.gradient(ffa_outline)[0]) + np.abs(np.gradient(ffa_outline)[1])
# outline[outline>0]=1
# outline[outline==0]=np.nan
# plt.imshow(eba_outline)
# plt.imshow(ofa_outline)
# fmri_nii=array_to_func1pt8(i,j,k,fmri_.detach().cpu(),faces_nii)
# fmri_surf=func1pt8_to_surfaces(fmri_nii,base+'/nsd/',sub,method='linear')


