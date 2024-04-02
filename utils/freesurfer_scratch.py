import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry



base_path='/home/uzair/nvme/nsd/'
sub='subj01'
t1_orig=nib.load(base_path+ '/freesurfer/%s/mri/orig/001.nii.gz'%sub)
t1=nib.load(base_path+'/freesurfer/%s/mri/T1.nii.gz'%sub)
t1_mgz=nib.load(base_path+'/freesurfer/%s/mri/T1.mgz'%sub)
pial=read_geometry(base_path+'/freesurfer/%s/surf/rh.pial'%sub)

#we will always stay in the world space when plotting.
#what is the FOV in world space?
affine=t1_orig.affine
box_start,box_end=affine@np.array([0,0,0,1]),\
                  affine@np.array([t1_orig.shape[0]-1,t1_orig.shape[1]-1,t1_orig.shape[2]-1,1])

#pick a slice in world coordinates
y0=0 #this is the slice
X=np.linspace(box_start[0],box_end[0],t1_orig.shape[0])
Z=np.linspace(box_start[2],box_end[2],t1_orig.shape[2])
XX,ZZ=np.meshgrid(X,Z,indexing='ij')
YY=np.ones_like(XX)*y0

#bring t1_orig to this slice
world_to_vox=np.linalg.inv(t1_orig.affine)
voxels=world_to_vox@np.column_stack([XX.flatten(),YY.flatten(),ZZ.flatten(),np.ones_like(XX.flatten())]).T
voxels=np.round(voxels[:-1]).astype(int)
t1_orig_slice=t1_orig.get_fdata()[voxels[0],voxels[1],voxels[2]].reshape(XX.shape)
plt.subplots()[1].pcolor(XX,ZZ,t1_orig_slice)

#bring t1 to this slice
world_to_vox=np.linalg.inv(t1.affine)
voxels=world_to_vox@np.column_stack([XX.flatten(),YY.flatten(),ZZ.flatten(),np.ones_like(XX.flatten())]).T
voxels=np.floor(voxels[:-1]).astype(int)
t1_slice=t1.get_fdata()[voxels[0],voxels[1],voxels[2]].reshape(XX.shape)
plt.subplots()[1].pcolor(XX,ZZ,t1_slice,cmap='gray')

#bring surface to this slice (this is the tricky part)
#what is the transform from world coordinates to vertex coordinates?
#move vertex to world
vertex_to_vox=np.linalg.inv(t1_mgz.header.get_vox2ras_tkr())
vox_to_world=t1_mgz.header.get_vox2ras()
vertex_to_world=vox_to_world@vertex_to_vox
vertices=pial[0]
vertices=vertex_to_world@np.column_stack([vertices,np.ones(vertices.shape[0])]).T
#extract points on slice
dy=0.25
vertices_slice=[]
for point in vertices.T:
    #print(point.shape)
    if np.abs(point[1])<dy:
        print(1)
        vertices_slice.append(np.asarray([point[0],point[2]]))
vertices_slice=np.row_stack(vertices_slice)
plt.plot(vertices_slice[:,0],vertices_slice[:,1],'.',color='orange')


#now lets bring beta to orig space and then sample onto vertices
from nsdcode.nsd_mapdata import NSDmapdata

# beta_nii=nib.load('/home/uzair/nvme/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/meanbeta.nii'
#                 '.gz')

beta_nii=nib.load('/home/uzair/nvme/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/prf-eccrois.nii.gz')

# beta_nii=nib.load('/home/uzair/nvme/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR'
#                   '/betas_session01.nii.gz')

nsdmapdata=NSDmapdata('/home/uzair/nvme/nsd/')
beta0pt8=nsdmapdata.fit(1,'func1pt8','anat0pt8',beta_nii.get_fdata()[:,:,:,0],interptype='cubic') #this should be in
# sample space as
# t1_orig

#load the pial hemispheres
hemis=['lh','rh']
pials=[read_geometry(base_path+'/freesurfer/%s/surf/%s.equi0.5.pial'%(sub,hemi)) for hemi in hemis]

#go from vertex to mean_beta and get the scalar
#make interpolator
from scipy.interpolate import RegularGridInterpolator
i=np.arange(0,beta0pt8.shape[0])
j=np.arange(0,beta0pt8.shape[1])
k=np.arange(0,beta0pt8.shape[2])
beta_interp=RegularGridInterpolator((i,j,k),beta0pt8)#,method='cubic')

vertex_to_vox=np.linalg.inv(t1_mgz.header.get_vox2ras_tkr())
vox_to_world=t1_mgz.header.get_vox2ras()
vertex_to_world=vox_to_world@vertex_to_vox
vertex_to_vox_orig=np.linalg.inv(t1_orig.affine)@vertex_to_world

beta_surfs=[]
for h,hemi in enumerate(hemis):
    print(h)
    vertices=pials[h][0]
    #vox_to_sample=vertex_to_vox_orig@()
    pts=(vertex_to_vox_orig@np.column_stack([vertices,np.ones(vertices.shape[0])]).T)[:-1]
    beta_surfs.append(beta_interp(pts.T))

from mayavi import mlab
#get the inflated
#load the pial hemispheres
hemis=['lh','rh']
inflates=[read_geometry(base_path+'/freesurfer/%s/surf/%s.inflated'%(sub,hemi)) for hemi in hemis]

mlab.triangular_mesh(inflates[0][0][:,0],inflates[0][0][:,1],inflates[0][0][:,2],inflates[0][1],scalars=beta_surfs[
    0],vmax=2000,vmin=-2000)

mlab.triangular_mesh(inflates[1][0][:,0]+100,inflates[1][0][:,1],inflates[1][0][:,2],inflates[1][1],scalars=beta_surfs[
    1],vmax=2000,vmin=-2000)
mlab.show()

