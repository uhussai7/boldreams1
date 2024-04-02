import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry
from cortex.freesurfer import get_surf,parse_surf,parse_patch
from utils import change_freesurfer_subjects
import cortex
from nsdcode.nsd_mapdata import NSDmapdata
from numpy.linalg import inv
import svgwrite
from matplotlib.colors import rgb2hex
from scipy.interpolate import RegularGridInterpolator
from mayavi import mlab
import cairosvg
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
import sys
#from dataHandling import dataFetcher
from utils.io_utils import load_config
from lucent.optvis import render
from boldreams.objectives import *
from models.fmri import prep_encoder
from boldreams import dream_wrapper
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render
from torch.optim import Adam,SGD
from models.fmri import prep_encoder
#sys.path.append('../')
#from surface import *
from torchvision import io
import random
from scipy.ndimage import binary_dilation,generate_binary_structure


def vertex_2_vox_anat_0pt8(anat_nii,t1_mgz_nii):
    ras_2_vox_anat_0pt8=inv(anat_nii.affine)
    vertex_2_ras=t1_mgz_nii.header.get_vox2ras() @ inv(t1_mgz_nii.header.get_vox2ras_tkr())
    return ras_2_vox_anat_0pt8 @ vertex_2_ras

def get_pials(freesurfer_path,subj):
    return read_geometry(freesurfer_path+'/%s/surf/lh.equi0.5.pial'%subj),\
           read_geometry(freesurfer_path+'/%s/surf/rh.equi0.5.pial'%subj)

def get_flats(freesurfer_path,subj):
    #subject need to be imported into pycortex
    change_freesurfer_subjects(freesurfer_path)
    flats_lh = get_surf(subj, 'lh', 'patch', 'full' + '.flat', freesurfer_subject_dir=freesurfer_path)
    flats_rh = get_surf(subj, 'rh', 'patch', 'full' + '.flat', freesurfer_subject_dir=freesurfer_path)
    return flats_lh,flats_rh

def make_interpolator_3d(nii,method='linear'):
    shp=nii.shape
    i,j,k=np.arange(0,shp[0]),np.arange(0,shp[1]),np.arange(0,shp[2])
    return RegularGridInterpolator((i,j,k),nii.get_fdata(),method=method,bounds_error=False)

def make_interpolator_3d_slice(nii,slice_ind=0,method='linear'):
    shp=nii.shape
    i,j,k=np.arange(0,shp[0]),np.arange(0,shp[1]),np.arange(0,shp[2])
    return RegularGridInterpolator((i,j,k),nii.get_fdata()[:,:,:,slice_ind],method=method,bounds_error=False)

def move_vertices(vertices,xfm):
    #vertices 3xN
    vertices=np.row_stack([vertices,np.ones([1,vertices.shape[-1]])])
    return (xfm @ vertices)[:-1]

def make_svg(hemis,scalars,path,r=0.8):
    #define some variables
    print('Working...')
    lh_pts = np.copy(hemis[0][0][:, :-1])
    rh_pts = np.copy(hemis[1][0][:, :-1])
    lh_sclr = scalars[0]
    rh_sclr = scalars[1]

    # offset the left hemisphere and flip the y-axis
    left_offset = 1.03 * (rh_pts[:, 0].max() - rh_pts[:, 0].min())
    lh_pts[:, 0] = lh_pts[:, 0] - left_offset
    lh_pts[:, 1], rh_pts[:, 1] = -lh_pts[:, 1], -rh_pts[:, 1]

    # get box dimensions
    u_min = np.min([lh_pts[:, 0].min(), rh_pts[:, 0].min()])  # u is horizontal dimension
    u_max = np.max([lh_pts[:, 0].max(), rh_pts[:, 0].max()])
    v_min = np.min([lh_pts[:, 1].min(), rh_pts[:, 1].min()])  # v is vertical dimension
    v_max = np.max([lh_pts[:, 1].max(), rh_pts[:, 1].max()])

    L = u_max-u_min
    H = v_max-v_min

    #aspect ratio and image dimensions
    aspect_ratio=L/H
    L_dim=600
    H_dim=L_dim/aspect_ratio
    s=L/L_dim


    #make a transform
    xfm = np.asarray([[1/s,0,abs(u_min)/s],
                      [0,1/s,abs(v_min)/s],
                      [0,0,1]])

    lh_pts_,rh_pts_=[move_vertices(p.T,xfm) for p in [lh_pts,rh_pts]]
    pts=np.column_stack([lh_pts_,rh_pts_])
    sclrs=np.concatenate([lh_sclr,rh_sclr])
    cmap = plt.get_cmap('jet')
    colors = cmap(sclrs / sclrs.max())

    dwg = svgwrite.Drawing(path, profile='tiny', size=(L_dim, H_dim))

    for i in range(0,len(pts.T),2):
        ut=pts[0,i]
        vt=pts[1,i]
        #print(ut,vt)
        dwg.add(dwg.circle(center=(ut, vt), r=r, fill=rgb2hex(colors[i,:-1])))
    dwg.save()
    #create a png for better loading
    cairosvg.svg2png(url=path,write_to=path.split('.')[0] + '.png',output_width=2*L_dim,output_height=2*H_dim)
    print('Done')

def func1pt8_to_surfaces(beta_nii,base_path,sub,method='nearest'):
    #base_path='/home/uzair/nvme/nsd/'
    #sub='subj01'
    #define all the paths
    t1_orig=nib.load(base_path+ '/freesurfer/%s/mri/orig/001.nii.gz'%sub)
    t1_mgz=nib.load(base_path+'/freesurfer/%s/mri/T1.mgz'%sub)
    freesurfer_path=base_path + '/freesurfer/'

    #map to anat
    nsdmapdata=NSDmapdata(base_path)
    beta0pt8=nsdmapdata.fit(1,'func1pt8','anat0pt8',beta_nii.get_fdata()[:,:,:],interptype=method) #this should be in
    beta0pt8_nii=nib.Nifti1Image(beta0pt8,t1_orig.affine)

    #get pial surfaces and move them to anat
    pials=get_pials(freesurfer_path,sub)
    xfm=vertex_2_vox_anat_0pt8(t1_orig,t1_mgz)
    vertices_vox=[move_vertices(hemi[0].T,xfm) for hemi in pials]

    #sample volume
    interp=make_interpolator_3d(beta0pt8_nii,method=method)
    scalars=[interp(pts.T) for pts in vertices_vox]

    return scalars,pials

def anat_to_surfaces(beta0pt8_nii,base_path,sub,method='nearest'):
    t1_orig = nib.load(base_path + '/freesurfer/%s/mri/orig/001.nii.gz' % sub)
    t1_mgz = nib.load(base_path + '/freesurfer/%s/mri/T1.mgz' % sub)
    freesurfer_path = base_path + '/freesurfer/'


    # get pial surfaces and move them to anat
    pials = get_pials(freesurfer_path, sub)
    xfm = vertex_2_vox_anat_0pt8(t1_orig, t1_mgz)
    vertices_vox = [move_vertices(hemi[0].T, xfm) for hemi in pials]

    # sample volume
    interp = make_interpolator_3d(beta0pt8_nii, method=method)
    scalars = [interp(pts.T) for pts in vertices_vox]

    return scalars, pials


def array_to_func1pt8(i,j,k,scalar,ref_nii):
    out=np.zeros_like(ref_nii.get_fdata())
    out[i,j,k]=scalar
    out=nib.Nifti1Image(out,ref_nii.affine)
    return out

def make_flat_map(scalars,flat_hemis,save_path,show_image=False,*args,**kwargs):
    # figure out the viewport and stuff and start making some flat maps
    # define some variables
    print('Working...')
    lh_pts = np.copy(flat_hemis[0][0])
    rh_pts = np.copy(flat_hemis[1][0])
    lh_edges=flat_hemis[0][1]
    rh_edges=flat_hemis[1][1]
    lh_sclr = scalars[0]
    rh_sclr = scalars[1]

    # offset the left hemisphere and flip the y-axis
    left_offset = 1.005 * (rh_pts[:, 0].max() - rh_pts[:, 0].min())
    lh_pts[:, 0] = lh_pts[:, 0] - left_offset
    #lh_pts[:, 1], rh_pts[:, 1] = -lh_pts[:, 1], -rh_pts[:, 1]

    # get box dimensions
    u_min = np.min([lh_pts[:, 0].min(), rh_pts[:, 0].min()])  # u is horizontal dimension
    u_max = np.max([lh_pts[:, 0].max(), rh_pts[:, 0].max()])
    v_min = np.min([lh_pts[:, 1].min(), rh_pts[:, 1].min()])  # v is vertical dimension
    v_max = np.max([lh_pts[:, 1].max(), rh_pts[:, 1].max()])

    L = u_max - u_min
    H = v_max - v_min

    if show_image==False:
        mlab.options.offscreen = True
    else:
        mlab.options.offscreen = False

    mlab.figure(size=(3*L, 3*H), bgcolor=(1, 1, 1))

    lh_=mlab.triangular_mesh(lh_pts[:,0],lh_pts[:,1],lh_pts[:,2],lh_edges,scalars=lh_sclr,*args,**kwargs)  # ,opacity=1,
    # vmax=6,
    # vmin=-2)
    rh_=mlab.triangular_mesh(rh_pts[:,0],rh_pts[:,1],rh_pts[:,2],rh_edges,scalars=rh_sclr,*args,**kwargs)

    change_lut(lh_)
    change_lut(rh_)

    mlab.draw()


    mlab.view(0, 0, 0.9*L)
    colorbar = mlab.colorbar()
    colorbar.label_text_property.color = (0, 0, 0)
    #colorbar.scalar_bar_representation.proportional_resize = True
    lower_left=[0.36, 0.02]
    thickness=0.06
    colorbar.scalar_bar_representation.position = lower_left
    colorbar.scalar_bar_representation.position2 = [(0.5-lower_left[0])*2, thickness]
    mlab.savefig(save_path)
    print('Done')

def change_lut(surf):
    cmap = plt.get_cmap('coolwarm')
    cmaplist = np.array([cmap(i) for i in range(cmap.N-1)]) * 255
    surf.module_manager.scalar_lut_manager.lut.table = cmaplist




def clamp(roi):
    roi[roi > 0] = 1
    return roi


class flat:
    def __init__(self,base,dic,ref_nii=None,sub='subj01'):
        #paths
        self.base = base
        self.freesurfer_path=self.base + '/nsd/freesurfer/'
        self.flat_path=self.base+'/dream_figures/'
        self.temp_path=self.flat_path+'/temp/'
        os.makedirs(self.flat_path,exist_ok=True)
        os.makedirs(self.temp_path,exist_ok=True)
        #dic
        self.dic = dic
        self.i,self.j,self.k=self.dic['x'], self.dic['y'], self.dic['z']
        #ref_nii
        if ref_nii is None:
            self.set_ref_nii()
        self.sub=sub
    def set_ref_nii(self):
        self.ref_nii = nib.load(self.base + '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')

    def project_from_array(self,array,file_name=None,vmin=None,vmax=None,method='nearest',*args,**kwargs): #dont really
        # need a
        # filename
        # here, use some temp name and then delete, maybe a random string.
        call_from_overlay=0
        if file_name is None:
            file_name=str(random.randint(10**9, (10**10)-1)) + '.png'
        else:
            call_from_overlay=1

        nii=array_to_func1pt8(self.i,self.j,self.k,array,self.ref_nii)
        nii_surf=func1pt8_to_surfaces(nii,self.base+'/nsd/',self.sub,method=method)[0]
        flat_hemis=get_flats(self.freesurfer_path,self.sub)
        if vmin is None:
            make_flat_map(nii_surf,flat_hemis,self.temp_path+file_name,*args,**kwargs)
        else:
            make_flat_map(nii_surf,flat_hemis,self.temp_path+file_name,
                          vmin=vmin,vmax=vmax,*args,**kwargs)
        if call_from_overlay==0:
            os.remove(self.temp_path+file_name)


    def project_from_array_overlay(self,array,overlay_file_names,out_filename='test.png',
                                   vmin=None,vmax=None,method='nearest',*args, **kwargs):
        file_name=str(random.randint(10**9, (10**10)-1)) + '.png'
        self.project_from_array(array,file_name,vmin,vmax,method,*args,**kwargs)
        acc = plt.imread(self.temp_path+file_name)
        H, L = acc.shape[:2]
        x_off = 355
        y_off = 200
        out_fig_name = self.flat_path + str(out_filename) #+ file_name.split('.png')[0]
        plt.ioff()
        fig, ax = plt.subplots()
        ax.imshow(acc[y_off:, int(L / 2) - x_off:int(L / 2) + x_off])
        ax.set_axis_off()
        for overlay_file_name in overlay_file_names:
            print('Loading: %s'%(overlay_file_name))
            overlay=plt.imread(self.base+'/overlays/'+overlay_file_name)
            if overlay_file_name.endswith('_outline.png'):
                overlay=change_overlay_color(overlay)
            ax.imshow(overlay[y_off:, int(L / 2) - x_off:int(L / 2) + x_off])
            #out_fig_name=out_fig_name + '_'+overlay_file_name
        print('Saving overlay in %s'%out_fig_name)
        plt.savefig(out_fig_name,dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        os.remove(self.temp_path+file_name)
        return ax

def change_overlay_color(overlay):
    alpha_channel=overlay[:,:,3]
    alpha_channel[alpha_channel>0]=True
    alpha_channel_= binary_dilation(alpha_channel,structure=np.ones((7, 7)), iterations=10)
    overlay[:,:,:3][alpha_channel_>0]=[0, 0, 0]#[247, 0, 255]
    return overlay
class DreamFlatmaps:
    def __init__(self,base_path,dic):
        self.dic=dic
        self.base_path=base_path
        self.dream_path=self.base_path+'/dreams/'
        self.overlays_path=self.base_path+'/overlays/'
        self.flatter=flat(self.base_path,self.dic)

    def make_dream_flatmap(self,model,bbtrain,roi,obj,overlay_file_names,out_filename='',max_channels='100',
                           *args, **kwargs):
        self.make_file_prefix(model,bbtrain,roi,obj,max_channels)
        self.out_filename=out_filename
        self.fmri=self.load_fmri()
        self.flatter.project_from_array_overlay(self.fmri,overlay_file_names,out_filename=out_filename,*args, **kwargs)

    def make_dream_ref_img_flatmap(self,model,bbtrain,roi,obj,overlay_file_names,out_filename='',max_channels='100',
                                   *args, **kwargs):
        self.make_file_prefix(model,bbtrain,roi,obj,max_channels,ref_img=True)
        self.out_filename=out_filename
        self.fmri=self.load_fmri()
        self.flatter.project_from_array_overlay(self.fmri,overlay_file_names,out_filename=out_filename,*args, **kwargs)


    def make_signal_flatmap(self,fmri_signal,overlay_file_names,out_filename='',*args, **kwargs):
        self.out_filename=out_filename
        self.flatter.project_from_array_overlay(fmri_signal,overlay_file_names,out_filename=out_filename,*args, **kwargs)

    def make_file_prefix(self,model,bbtrain,roi,obj,max_channels,ref_img=False):
        if ref_img==True:
            self.file_prefix = '/ref_img/bb-%s_upto-16_bbtrain-%s_max-channels_%s_roi-%s_obj-%s' % (model,
                                                                                                    bbtrain,
                                                                                                    max_channels,
                                                                                                    roi,
                                                                                                    obj)
        else:
            self.file_prefix='bb-%s_upto-16_bbtrain-%s_max-channels_%s_roi-%s_obj-%s'%(model,bbtrain,max_channels,
                                                                                       roi,obj)

    def load_fmri(self):
        fmri_file=self.file_prefix+'_fmri.pt'
        return torch.load(self.dream_path+ fmri_file)

    def load_img(self):
        img_file=self.file_prefix+'_img.png'
        return io.read_image(self.dream_path+ img_file)

    def figure_img_flat_hist(self):
        img = self.load_img()
        flat_img = plt.imread(self.base_path + '/dream_figures/' + self.out_filename)

        fig = plt.figure(figsize=(17, 12))

        box_flat = [0.35, 0, 0.6, 1]
        box_img = [0.05, 0, 0.28, 1]
        #box_hist = [0, 0, 0.5, 0.5] #might have to add this later
        #ax_flat, ax_img, ax_hist = fig.add_axes(box_flat), fig.add_axes(box_img), fig.add_axes(box_hist)
        ax_flat, ax_img = fig.add_axes(box_flat), fig.add_axes(box_img)
        ax_flat.imshow(flat_img)
        ax_flat.set_axis_off()
        ax_img.set_axis_off()
        ax_img.imshow(img.moveaxis(0, -1))
        plt.title(self.out_filename)
        plt.show()

    def overlays_to_draw(self,fmri_mean_per_roi):
        #how do we determine which overlays are to be drawn?
        fmri_mean_per_roi = self.fmri_roi_mean(self.fmri)

    def fmri_roi_mean(self,fmri):
        #this is quite annoying because some the ROIs have been combined in the outline
        #we need to define a new dictionary with roifilters for the rois for which we have overlays.
        #for the basic rois the mapping from roi to overlay is the same
        roi_dic_c=self.dic['roi_dic_combined']
        roi_dic=self.fic['roi_dic']
        overlay_rois={'V1V2V3': clamp(roi_dic_c['V1'] + roi_dic['V2'] + roi_dic['V3']),
                      'V3ab': clamp(roi_dic_c['V3ab']),
                      'hV4':clamp(roi_dic_c['hV4']),
                      'VO': clamp(roi_dic_c['VO']),
                      'LO':clamp(roi_dic_c['LO']),
                      'PHC':clamp(roi_dic_c['PHC']),
                      'IPS':clamp(roi_dic_c['IPS']),
                      'MT':clamp(roi_dic_c['MT']),
                      'MST':clamp(roi_dic_c['MST']),
                      'floc-bodies_EBA': clamp(roi_dic['EBA']),
                      'floc-bodies_FBA': clamp(roi_dic['FBA-1']+roi_dic['FBA-2']),
                      'floc-faces_FFA': clamp(roi_dic['FFA-1']+roi_dic['FFA-2']),
                      'floc-faces_OFA': clamp(roi_dic['OFA']),
                      'floc_faces_aTL-faces':clamp(roi_dic['aTL-faces']),
                      'floc-places_PPA':clamp(roi_dic['PPA']),
                      'floc-places_OPA':clamp(roi_dic['OPA']),
                      'floc-places_RSC':clamp(roi_dic['RSC']),
                      }
        fmri_mean_per_roi={}
        for key in overlay_rois.keys():
            roi_filter=overlay_rois[key]
            fmri_mean_per_roi[key]=fmri[roi_filter].mean()
        return fmri_mean_per_roi
