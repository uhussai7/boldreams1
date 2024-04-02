import numpy as np

from utils.surface import *
from mayavi import mlab
import sys
from dataHandling import dataFetcher
import nibabel as nib
from utils.roi_utils import anat_combine_rois,vol_for_flat_maps

#loading data
sys.argv=sysargs(sys.argv,'RN50x4_clip_relu3_last',1,10,-1,'False')
#sys.argv=sysargs(sys.argv,'alexnet',6,10,-1,'False')

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
rois=df.brain_and_floc_rois()

subj='subj01'
roi_list=['prf-eccrois']
test,roi_inds=vol_for_flat_maps(roi_list,config['base_path'],subj)
roi_vol=NiftiVolume(nii=test)
# #base_path='/home/uzair/nvme/nsd/'
freesurfer_path=config['base_path'] + '/nsd/freesurfer/' #+ '/' + subj + '/'
# func1pt8toanat=NiftiVolume(config['base_path'] + '/nsd/nsddata/ppdata/' + subj +
#                            '/transforms/func1pt8-to-anat0pt8.nii.gz')
#
# #train data
# imgs,fmri=df.training_data_loader().dataset.tensors
#
# #get the roi voxel
# mask_nii=anat_combine_rois(config['base_path'] + '/nsd/nsddata/ppdata/' + subj + '/anat/roi/')
#
# #
# mask_vol=NiftiVolume(nii=mask_nii,method='nearest')
# faces_vol=NiftiVolume(config['base_path'] + '/nsd/nsddata/ppdata/' + subj + '/anat/roi/floc-faces.nii.gz',
#                       method='nearest')
#
flat=FlatSurface(0,freesurfer_path,subj,'lh')
roi_flat=flat.project_vol(roi_vol)
#faces=flat.project_vol(faces_vol)
#test=flat.project_fmri_list(fmri[0],df.x,df.y,df.z)
#flat.make_rois_curves(rois,df.x,df.y,df.z)

flat1=FlatSurface(0,freesurfer_path,subj,'rh')
roi_flat1=flat1.project_vol(roi_vol)



vol_ind=200
mlab.triangular_mesh(flat.fpts[:,0],flat.fpts[:,1],flat.fpts[:,2],flat.fpolys,scalars=roi_flat)
mlab.triangular_mesh(flat1.fpts[:,0]+350,flat1.fpts[:,1],flat1.fpts[:,2],flat1.fpolys,scalars=roi_flat1)
                     #vmin=-2,vmax=10)
mlab.view(0.0, 0.0, 1200)
mlab.colorbar()