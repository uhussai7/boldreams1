import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator,NearestNDInterpolator,LinearNDInterpolator
from utils import *
import cortex
from cortex.freesurfer import get_surf
import struct
from .surface_utils import array_to_func1pt8,func1pt8_to_surfaces, make_flat_map,get_flats
from .io_utils import get_base_path
import matplotlib.pyplot as plt
import random
import os
from scipy.ndimage import binary_dilation,generate_binary_structure
#need something to readily make flat images


# def parse_surf(filename):
#     """
#     """
#     with open(filename, 'rb') as fp:
#         #skip magic
#         fp.seek(3)
#         comment = fp.readline()
#         fp.readline()
#         print(comment)
#         verts, faces = struct.unpack('>2I', fp.read(8))
#         pts = np.fromstring(fp.read(4*3*verts), dtype='f4').byteswap()
#         polys = np.fromstring(fp.read(4*3*faces), dtype='i4').byteswap()
#
#         return pts.reshape(-1, 3), polys.reshape(-1, 3)
#
# class NiftiVolume:
#     def __init__(self,file_path=None,nii=None,bounds_error=False,method='linear'):
#         if file_path is not None:
#             self.file_path=file_path
#             self.nii=nib.load(self.file_path)
#         else:
#             self.nii=nii
#         self.sform=self.nii.get_sform()
#         self.bounds_error=bounds_error
#         self.method=method
#         self.make_interpolator()
#
#     def make_interpolator(self):
#         self.interpolator=RegularGridInterpolator([np.arange(0,self.nii.shape[a]) for a in range(0,3)],
#                                                   self.nii.get_fdata(),bounds_error=self.bounds_error,
#                                                   method=self.method)
#     def __call__(self, point,*args, **kwargs):
#         return self.interpolator(point)
#
# class FlatBrain:
#     def __init__(self,file_path,freesurfer_path,subj,xfm_vol=None):
#         hemis=['lh','rh']
#         self.flat_surfs=[FlatSurface(file_path,freesurfer_path,subj,hemi,xfm_vol) for hemi in hemis]
#     def project_fmri_vol(self,fmri,xfm_vol=None):
#         fmris=[f.project_fmri_vol(fmri,xfm_vol) for f in self.flat_surfs]
#         return fmris
#     def project_fmri_list(self,fmri,x,y,z,xfm_vol=None,bounds_error=True,method='linear'):
#         fmris=[f.project_fmri_list() for f in self.flat_surfs]
#         return fmris
#
# class FlatSurface:
#     def __init__(self,file_path,freesurfer_path,subj,hemi,xfm_vol=None):
#         self.file_path=file_path
#         self.freesurfer_path=freesurfer_path
#         self.header_path=self.freesurfer_path+subj+'/mri/T1.mgz'
#         self.header=nib.load(self.header_path).header
#         self.Torig=self.header.get_vox2ras_tkr()
#         self.Norig=np.asmatrix(self.header.get_vox2ras())
#         change_freesurfer_subjects(freesurfer_path)
#         #cortex.freesurfer.import_subj(subj,freesurfer_subject_dir=freesurfer_path)
#         #cortex.freesurfer.import_flat(subj,'full',clean=False,auto_overwrite=True)
#         self.fpts, self.fpolys, _ = get_surf(subj, hemi, 'patch', 'full' + '.flat',
#                                        freesurfer_subject_dir=freesurfer_path)
#         self.pts,self.polys=nib.freesurfer.read_geometry(freesurfer_path+subj+'/surf/'+hemi+'.equi0.5.pial')
#         if xfm_vol is not None:
#             self.init_xfm_vol(xfm_vol)
#
#     def init_xfm_vol(self,xfm_vol):
#         self.xfm_vol = xfm_vol
#         self.sform = xfm_vol.sform
#         self.to_xfm_vol = np.matmul(np.linalg.inv(self.sform), np.matmul(self.Norig, np.linalg.inv(self.Torig)))
#         self.coords = np.append(self.pts, np.ones((self.pts.shape[0], 1)), axis=1)
#         self.coords_ = np.asarray(np.matmul(self.to_xfm_vol, self.coords.T).T)
#         self.coords__ = xfm_vol(self.coords_[:, :3])
#
#     def project_fmri_vol(self,fmri,xfm_vol=None):
#         #fmri should be a class of nitivol
#         #first we get position in xfm_vol
#         if xfm_vol is not None:
#             self.init_xfm_vol(xfm_vol)
#         #now sample fmri at these coords
#         return fmri(self.coords__[:,:3])
#
#     def project_fmri_list(self,fmri,x,y,z,xfm_vol=None,bounds_error=True,method='linear'):
#         if xfm_vol is not None:
#             self.init_xfm_vol(xfm_vol)
#         return LinearNDInterpolator((x,y,z),fmri)(self.coords__)
#
#     def project_vol(self,vol):
#         sform=vol.sform
#         to_vol_voxels = np.matmul(np.linalg.inv(sform), np.matmul(self.Norig, np.linalg.inv(self.Torig)))
#         coords = np.append(self.pts, np.ones((self.pts.shape[0], 1)), axis=1)
#         coords_ = np.asarray(np.matmul(to_vol_voxels, coords.T).T)
#         return vol(coords_[:,:3])
#
#     def make_rois_curves(self,rois,x,y,z):
#         # pts=NearestNDInterpolator((x, y, z), np.asarray(rois['hV4']))(self.coords__)
#         # mlab.triangular_mesh(self.fpts[:, 0], self.fpts[:, 1], self.fpts[:, 2], self.fpolys, scalars=pts.astype(float))
#         pass


# subj='subj01'
# hemi='lh'
# kind='pial'
# base_path='/home/uzair/nvme/nsd/'
# freesurfer_path=base_path + '/freesurfer/' + '/' + subj + '/'
# header_path=freesurfer_path + '/mri/T1.mgz'
# surface_path=freesurfer_path + '/surf/' + hemi + '.' + kind
#
# coords,faces=nib.freesurfer.io.read_geometry(surface_path)
# header=nib.load(header_path).header
#
# Torig=header.get_vox2ras_tkr()
# Norig = np.asmatrix(header.get_vox2ras())
#
# #load volume to project
# vol=nib.load(freesurfer_path+'/mri/orig/001.nii.gz')
# sform = vol.get_sform()
#
# xfm = np.matmul(Norig, np.linalg.inv(Torig))
# xfm = np.matmul(np.linalg.inv(sform), xfm)
#
# coords=np.append(coords,np.ones((coords.shape[0],1)),axis=1)
#
# coords_=np.matmul(xfm,coords.T).T
#
# i,j,k=[np.arange(0,vol.shape[a]) for a in range(0,3)]
#
#
# interpolator=RegularGridInterpolator((i,j,k),vol.get_fdata(),method='nearest')
#
# point=np.array((-32.23, 0.00, 42.25,1))
# interpolator(np.matmul(xfm,point)[0,:3])
#
# #load the flat surface
# import cortex
# from cortex.freesurfer import get_surf
# freesurfer_path='/home/uzair/nvme/nsd/freesurfer/'
# change_freesurfer_subjects(freesurfer_path)
# cortex.freesurfer.import_subj("subj01",freesurfer_subject_dir=freesurfer_path)
# cortex.freesurfer.import_flat("subj01",'full',clean=False,auto_overwrite=True)
# pts_lh,polys_lh,_=get_surf('subj01','lh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path)
# pts_rh,polys_rh,_=get_surf('subj01','rh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path)
#
# #okay now do all the neccessry transform to get the date to flat surface
# xfm_nii=nib.load(base_path + '/nsddata/ppdata/' + subj + '/transforms/func1pt8-to-anat1pt0.nii.gz')
# test=xfm_nii.get_fdata()
# test_interpolator= RegularGridInterpolator([np.arange(0,test.shape[a]) for a in range(0,3)],test)
