import numpy as np
import torch
import nibabel as nib
from src.file_utility import load_mask_from_nii, view_data
from collections import OrderedDict


def add_prefix_to_keys(dic,prefix='roi_'):
    out=OrderedDict()
    for key in list(dic.keys()):
        out[prefix+key]=dic[key]
    return out

def basic_roi_list():
    return ['floc-bodies','floc-faces','floc-places','floc-words','prf-eccrois']

def read_ctab(path):
    int_list=[]
    string_list=[]
    with open(path,'r') as file:
        for line in file:
            parts=line.strip().split(' ')
            if len(parts)==2:
                integer=int(parts[0])
                string=parts[1]
                if string != 'Unknown':
                    int_list.append(integer)
                    string_list.append(string)
    return int_list,string_list

def update_mask(mask,rois,subj,mask_root):
    for roi in rois:
        path=mask_root+'subj%02d/func1pt8mm/roi/'%subj +'/'+roi+'.nii.gz'
        mask[load_mask_from_nii(path)>0]=1
    return mask

def vol_for_flat_maps(roi_list,root,subj):
    #set the paths
    freesurfer_path=root + '/nsd/freesurfer/'+subj+'/'
    anat_path=root+'/nsd/nsddata/ppdata/' + subj + '/anat/roi/'
    #first sort outt he base
    kast_nii=nib.load(anat_path+'Kastner2015.nii.gz')
    prf=nib.load(anat_path+'prf-visualrois.nii.gz').get_fdata()
    rois=kast_nii.get_fdata()
    rois[prf>0]=prf[prf>0]
    rois[rois<1]=0

    #make a dictionary for all the labels
    roi_inds={}
    inds,keys=read_ctab(freesurfer_path+'/label/%s.mgz.ctab'%('Kastner2015'))
    for k,key in enumerate(keys):
        roi_inds[key]=inds[k]
    max_index=rois.max()

    #take care of all the other rois
    for roi in roi_list:
        roi_vol=nib.load(anat_path+roi+'.nii.gz').get_fdata()
        roi_vol=roi_vol+max_index
        inds,keys=read_ctab(freesurfer_path+'/label/%s.mgz.ctab'%(roi))
        for k,key in enumerate(keys):
            rois[roi_vol==inds[k]+max_index]=roi_vol[roi_vol==inds[k]+max_index]
            roi_inds[key]=int(inds[k]+max_index)
        max_index=rois.max()
    return nib.Nifti1Image(rois,kast_nii.affine),roi_inds


def make_roi_dic(mask,roi_dic,rois,subj,mask_root,nsd_root):
    mask=mask.flatten()
    for roi in rois:
        roi_nii=load_mask_from_nii(mask_root+'subj%02d/func1pt8mm/roi/'%subj +'/'+roi+'.nii.gz').flatten()
        if roi == 'basic_rois':
            inds,keys=read_ctab(nsd_root + '/freesurfer/subj%02d/label/%s.mgz.ctab'%(subj,'Kastner2015'))
        else:
            inds, keys = read_ctab(nsd_root + '/freesurfer/subj%02d/label/%s.mgz.ctab' % (subj, roi))
        for k,key in enumerate(keys):
            roi_dic[key]=roi_nii[mask==1]==inds[k]
    return roi_dic

def combine_rois(roi_dic,roi_list,nsd_root,subj):
    roi_dic_out={}
    basic_grouping={'V1':['V1v','V1d'],'V2':['V2v','V2d'],'V3':['V3v','V3d'],'V3ab':['V3A','V3B'],
                    'hV4':['hV4'], 'VO':['VO1','VO2'],
                    'LO': ['LO1', 'LO2'],
                    'PHC':['PHC1','PHC2'],
                    'IPS':['IPS0','IPS1','IPS2','IPS3','IPS4','IPS5'],
                    'MT':['TO1'],'MST':['TO2']}
    for key in basic_grouping:
        roi_dic_out[key]=combine_filters([roi_dic[k] for k in basic_grouping[key]])
    for roi in roi_list:
        inds, keys = read_ctab(nsd_root + '/freesurfer/subj%02d/label/%s.mgz.ctab' % (subj, roi))
        roi_dic_out[roi]=combine_filters([roi_dic[k] for k in keys])
    return roi_dic_out

def combine_filters(filters):
    out=np.zeros_like(filters[0]).astype(bool)
    for filter in filters:
        out[filter]=1
    return out

def roi_from_nii(nii,i,j,k,ctab):
    """
    We need more finer ROIs for the floc regions
    :param nii: nifti file path
    :param i,j,k: these are the indices to sample the nifti at
    :param ctab: table for names of finer rois
    :return: roi dic
    """
    nii=nib.load(nii)
    rois=nii.get_fdata()[i,j,k]
    vals,keys=read_ctab(ctab)
    out={}
    for key,val in zip(keys,vals):
        out[key]=rois==val
    return out


def anat_combine_rois(mask_root):
    prf_nii = nib.load(mask_root + "/prf-visualrois.nii.gz")
    kast_nii = nib.load(mask_root + "/Kastner2015.nii.gz")
    prf_mask=prf_nii.get_fdata()>0

    prf_array=prf_nii.get_fdata()
    kast_array=kast_nii.get_fdata()

    kast_array[prf_mask]=prf_array[prf_mask]

    return nib.Nifti1Image(kast_array,kast_nii.affine)


def remove_keys(roi_dic, keywords=['rh', 'lh']):
    roi_dic_out = {}
    if not isinstance(keywords,list):
        keywords=[keywords]
    for key in roi_dic.keys():
        if not any(keyword in key for keyword in keywords):
            roi_dic_out[key] = roi_dic[key]
    return roi_dic_out

def greater_than_zero(roi_dic):
    roi_dic_out={}
    for key,value in roi_dic.items():
        roi_dic_out[key]=value>0
    return roi_dic_out

def flat_to_vol(df,fmri,shape=(83,104,81)):
    out=np.zeros(shape)
    x=df.dic['x'][df.dic['viscort_roi']]
    y=df.dic['y'][df.dic['viscort_roi']]
    z=df.dic['z'][df.dic['viscort_roi']]
    out[z,y,x]=fmri
    return out
def get_overlay_file_names():
    overlay_file_names= {'floc-faces':['floc-faces_FFA_outline.png','floc-faces_FFA_outline_label.png',
                                   'floc-faces_OFA_outline.png','floc-faces_OFA_outline_label.png',
                                   'floc-bodies_EBA_outline.png','floc-bodies_EBA_outline_label.png'],
                     'floc-bodies':['floc-bodies_EBA_outline.png','floc-bodies_EBA_outline_label.png',
                                    'floc-bodies_FBA_outline.png','floc-bodies_FBA_outline_label.png'],
                     'floc-places':['floc-places_PPA_outline.png','floc-places_PPA_outline_label.png',
                                    'floc-places_OPA_outline.png','floc-places_OPA_outline_label.png',
                                    'floc-places_RSC_outline.png','floc-places_RSC_outline_label.png'],
                     'V1':['V1V2V3_outline.png','V1V2V3_outline_label.png'],
                     'V2':['V1V2V3_outline.png','V1V2V3_outline_label.png'],
                     'V3':['V1V2V3_outline.png','V1V2V3_outline_label.png'],
                     'hV4':['hV4_outline.png','hV4_outline_label.png'],
                     'V3ab':['V3ab_outline.png','V3ab_outline_label.png'],
                     'VO':['VO_outline.png','VO_outline_label.png'],
                     'LO':['LO_outline.png','LO_outline_label.png'],
                     'PHC':['PHC_outline.png','PHC_outline_label.png'],
                     'IPS':['IPS_outline.png','IPS_outline_label.png'],
                     'MT':['MT_outline.png','MT_outline_label.png'],
                     'MST':['MST_outline.png','MST_outline_label.png']}
    return overlay_file_names
#from .model_utils import hook_model

# def make_roi_filter_from_groups(group_names, group_vals, atlas):
#     """
#     this returns filters from group of rois combined together and one roi atlas
#     :param group_names: List of group names
#     :param group_vals: Labels that belong to each group
#     :param atlas: Atlas of labels
#     :return: dictionary of group names and rois
#     """
#     roi_filters= [torch.asarray([a in g for a in atlas]) for g in group_vals]
#     return {group_names[i] : roi_filters[i] for i in range(0,len(group_names))}
#
# def max_activation_ordering(imgs,model,roi,batch_size=3):
#     """
#     this returns the ordering for imgs such that the mean fmri activation in roi is in descending order
#     :param imgs: Images of shape [B,3,H,W]
#     :param model: Wrapped encoding model
#     :param roi: Roi for activation
#     :param batch_size: batch_size for forward pass for model
#     :return: ordered inds for the batch dimension
#     """
#     hook=hook_model(model.eval())
#     _=model(imgs[0].cuda())
#     Nv=hook(roi).shape[-1]
#     out=torch.zeros([len(imgs),Nv])
#     for i in range(0,len(imgs),batch_size):
#         _=model(imgs[i:i + batch_size].cuda())#.detach().cpu()
#         out[i:i+batch_size]=hook(roi).detach().cpu()
#     out=out.mean(-1)
#     return out.argsort(descending=True)

