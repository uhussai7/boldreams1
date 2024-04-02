import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry
from cortex.freesurfer import get_surf,parse_surf,parse_patch
from utils import change_freesurfer_subjects
import cortex
from nsdcode.nsd_mapdata import NSDmapdata

base_path='/home/uzair/nvme/nsd/'
sub='subj01'
t1_orig=nib.load(base_path+ '/freesurfer/%s/mri/orig/001.nii.gz'%sub)
orig_mgz=nib.load(base_path+ '/freesurfer/%s/mri/orig.mgz'%sub)
t1=nib.load(base_path+'/freesurfer/%s/mri/T1.nii.gz'%sub)
t1_mgz=nib.load(base_path+'/freesurfer/%s/mri/T1.mgz'%sub)
pial=read_geometry(base_path+'/freesurfer/%s/surf/lh.equi0.5.pial'%sub)

#load these flat maps
freesurfer_path='/home/uzair/nvme/nsd/freesurfer/'
change_freesurfer_subjects(freesurfer_path)
cortex.freesurfer.import_subj("subj01",freesurfer_subject_dir=freesurfer_path)
cortex.freesurfer.import_flat("subj01",'full',clean=False,auto_overwrite=True)

#prepare volume in anat space
beta_nii=nib.load('/home/uzair/nvme/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/prf-eccrois.nii.gz')
nsdmapdata=NSDmapdata('/home/uzair/nvme/nsd/')
beta0pt8=nsdmapdata.fit(1,'func1pt8','anat0pt8',beta_nii.get_fdata()[:,:,:],interptype='nearest') #this should be in
beta0pt8_nii=nib.Nifti1Image(beta0pt8,t1_orig.affine)
nib.save(beta0pt8_nii,'/home/uzair/test.nii.gz')

print('done with mapping')

#sort out the xfm
#lets just use identity
from cortex.xfm import Transform
vox2ras=t1.affine
vox2ras_tkr=t1_mgz.header.get_vox2ras_tkr()
vox2world_orig=t1_orig.affine
#we want tkr-world to world
xfm=vox2ras@np.linalg.inv(vox2ras_tkr)
xfm=Transform(xfm,beta0pt8_nii)
xfm.save('subj01','test2')

voxel_vol = cortex.Volume(beta0pt8_nii.get_fdata(), 'subj01', 'test2')

cortex.quickshow(voxel_vol)
plt.show()

patch=parse_patch(base_path+ '/freesurfer/%s/surf/lh.full.flat.patch.3d'%sub)
verts_lh = patch[patch['vert'] > 0]['vert'] - 1
edges_lh = -patch[patch['vert'] < 0]['vert'] - 1
patch=parse_patch(base_path+ '/freesurfer/%s/surf/rh.full.flat.patch.3d'%sub)
verts_rh = patch[patch['vert'] > 0]['vert'] - 1
edges_rh = -patch[patch['vert'] < 0]['vert'] - 1

flats_lh=get_surf('subj01','lh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path)
flats_rh=get_surf('subj01','rh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path)

from mayavi import  mlab

mlab.triangular_mesh(flats_lh[0][:,0]-500,flats_lh[0][:,1],flats_lh[0][:,2],flats_lh[1])
mlab.triangular_mesh(flats_rh[0][:,0],flats_rh[0][:,1],flats_rh[0][:,2],flats_rh[1])

#why cant we take these vertex points and sample anat
from scipy.interpolate import RegularGridInterpolator
interp=RegularGridInterpolator((np.arange(0,beta0pt8.shape[0]),
                                np.arange(0,beta0pt8.shape[0]),
                                np.arange(0,beta0pt8.shape[0])),beta0pt8,method='nearest',bounds_error=False)

from numpy.linalg import inv
vertex2func=inv(beta0pt8_nii.affine) @ vox2ras @ inv(vox2ras_tkr)
scalars=[]
for i in range(0,len(flats_lh[0])):
    pt=vertex2func @ np.append(pial[0][i],1)
    scalars.append(interp(pt[:-1]))
scalars=np.asarray(scalars)
mlab.triangular_mesh(flats_lh[0][:,0]-500,flats_lh[0][:,1],flats_lh[0][:,2],flats_lh[1],scalars=scalars[:,0])

import svgwrite
from matplotlib.colors import rgb2hex

cmap = plt.get_cmap('jet')
colors=cmap(scalars[:,0]/scalars.max())

dwg = svgwrite.Drawing("/home/uzair/nvme/test.svg", profile='tiny', size=(300, 300.1))
for i in range(0,len(flats_lh[0])):
    x,y=(flats_lh[0][i,0]+flats_lh[0][i,0].min())/2,(flats_lh[0][i,1] + flats_lh[0][i,1].min())/2
    dwg.add(dwg.circle(center=(x, y), r=0.3, fill=rgb2hex(colors[i,:-1])))

dwg.save()
