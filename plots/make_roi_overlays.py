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

#get config
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
df=dataFetcher(config['base_path'])
#get data
df.fetch(upto=1)#config['UPTO'])
sub='subj01'
freesurfer_path = base + '/nsd/freesurfer/'
anat_roi_path=base+ '/nsd/nsddata/ppdata/' + sub + '/anat/roi/'

#get the flat hemispheres
flat_hemis = get_flats(freesurfer_path, sub)

# #lets define overlays from anat space
# vals,keys=read_ctab(freesurfer_path+sub+'/label/floc-bodies.mgz.ctab') #get labels
# roi_niis = [nib.load(anat_roi_path + hemi + '.floc-bodies.nii.gz') for hemi in ['lh','rh']]
# for key,val in zip(keys,vals):
#     anat_nii=np.zeros_like(roi_niis[0].get_fdata())
#     for i in range(0,len(roi_niis)):
#         anat_nii[roi_niis[i].get_fdata()==val]=1
#     roi_surf=anat_to_surfaces(nib.Nifti1Image(anat_nii,roi_niis[0].affine),base+'/nsd/',sub,method='nearest')[0]
#     make_flat_map(roi_surf,flat_hemis,'/home/uzair/nvme/overlays1/%s.png'%key,show_image=False)


#make overlays for the basic rois
i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
roi_dic=df.dic['roi_dic_combined']
ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
for key in roi_dic.keys():
    print(key)
    roi_nii=array_to_func1pt8(i,j,k,roi_dic[key],ref_nii)
    roi_surf=func1pt8_to_surfaces(roi_nii,base+'/nsd/',sub)[0]
    make_flat_map(roi_surf,flat_hemis,'/home/uzair/nvme/overlays/%s.png'%key)



# #get finer rois
rois_dic=df.dic['roi_dic_combined']
i,j,k=df.dic['x'],df.dic['y'],df.dic['z']

#roi_name='floc-bodies'
for roi_name in ['floc-faces','floc-bodies','floc-places']:
# overlay_rois=roi_from_nii(base+'/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/lh.floc-faces.nii.gz',i,j,k,
#              freesurfer_path + sub + '/label/floc-faces.mgz.ctab')


    overlay_rois=roi_from_nii(base+'/nsd/nsddata/ppdata/subj01/anat/roi/lh.%s.nii.gz'%roi_name,i,j,k,
                 freesurfer_path + sub + '/label/%s.mgz.ctab'%roi_name)
    vals,keys=read_ctab(freesurfer_path+sub+'/label/%s.mgz.ctab'%roi_name) #get labels

    #we have to map rois to surfaces and then try to make an outline in svg
    ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
    #for key in overlay_rois.keys():
    for key,val in zip(keys,vals):
        #faces_nii=array_to_func1pt8(i,j,k,overlay_rois[key],ref_nii)
        faces_lh_nii=nib.load(base+'/nsd/nsddata/ppdata/subj01/anat/roi/lh.%s.nii.gz'%roi_name)
        faces_rh_nii=nib.load(base+'/nsd/nsddata/ppdata/subj01/anat/roi/rh.%s.nii.gz'%roi_name)

        nii=np.zeros_like(faces_lh_nii.get_fdata())
        #print((faces_nii.get_fdata==i))
        nii[faces_lh_nii.get_fdata()==val]=1
        nii[faces_rh_nii.get_fdata()==val]=1

        faces_surf=anat_to_surfaces(nib.Nifti1Image(nii,faces_lh_nii.affine),base+'/nsd/',sub,method='nearest')[0]
        make_flat_map(faces_surf,flat_hemis,'/home/uzair/nvme/overlays/%s_%s.png'%(roi_name,key))
