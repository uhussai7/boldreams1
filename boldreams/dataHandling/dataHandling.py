import os
import nibabel as nib
from scipy.io import loadmat
import h5py
import numpy as np
from src.file_utility import load_mask_from_nii, view_data
from src.roi import roi_map, iterate_roi
from src.load_nsd import load_betas
from src.load_nsd import image_feature_fn, data_split
from models.alexnet import Alexnet_fmaps
import torch
from src.torch_feature_space import filter_dnn_feature_maps
from torch.utils.data import DataLoader,TensorDataset
from collections import namedtuple
from torchvision.transforms import Resize
from utils import remove_keys,greater_than_zero,read_ctab,update_mask,make_roi_dic
from scipy.ndimage import morphology
from utils.roi_utils import combine_rois

def signal_in_roi(fmri,orig_roi,dic):
    groups=dic['group']
    fmri_out=[]
    for g in groups:
        this_filter=torch.asarray([a in g for a in orig_roi])
        fmri_out.append(fmri[this_filter])
    return fmri_out

def train_stimuli_from_dic(df,key='train_stim'):
    return torch.from_numpy(df.dic[key])

class dataFetcher:
    """
    A simple data fetcher
    """
    def __init__(self, base_path="/home/uzair/nvme/"):
        """
        Handle all paths in initializer
        :param base_path: path for nsd data
        """
        self.base_path=base_path
        self.nsd_root=self.base_path + '/nsd/'
        self.beta_folder= self.nsd_root + '/nsddata_betas/ppdata/'
        self.func_res='/func1pt8mm/'
        self.mask_root=self.nsd_root + '/nsddata/ppdata/'
        self.exp_design_file=self.nsd_root + '/nsddata/experiments/nsd/nsd_expdesign.mat'
        self.load_ext='.nii.gz'
        self.stim_folder=self.nsd_root + 'nsddata_stimuli/stimuli/nsd/'
        self.manual_roi_folder=self.nsd_root + '/nsddata/ppdata/subj01/func1pt8mm/roi/'
        self.exp_roi_folder=self.nsd_root + '/nsddata/ppdata/subj01/func1pt8mm/'

    def fetch(self,subj=1,upto=4,zscore=True,fmap_max=512,roi_list=['floc-bodies','floc-faces',
                                                                    'floc-places','floc-words',
                                                                    'prf-eccrois']):
        """
        Function to fetch data (relies mostly on styves nsd repo)
        :param subj:
        :param upto:
        :return:
        """
        #load experiment design file
        exp_design=loadmat(self.exp_design_file)
        ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

        #load stimuli
        image_data_set = h5py.File(self.stim_folder + "S%d_stimuli_227.h5py" % subj, 'r')
        image_data = np.copy(image_data_set['stimuli'])
        print('block size:', image_data.shape, ', dtype:', image_data.dtype, ', value range:', \
              np.min(image_data[0]), np.max(image_data[0]))

        #rois
        group_names = ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST','other']
        group = [[1, 2], [3, 4], [5, 6], [7], [16, 17], [14, 15], [18, 19, 20, 21, 22, 23], [8, 9], [10, 11], [13],
                 [12], [24,25, 0]]


        #load rois
        basic_rois=nib.load(self.mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % subj)
        affine=basic_rois.affine
        basic_rois=basic_rois.get_fdata()
        prf=load_mask_from_nii(self.mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz" % subj)
        basic_rois[prf>0]=prf[prf>0] #this is step that the original others do

        voxel_mask_full = load_mask_from_nii(self.mask_root + "subj%02d/func1pt8mm/brainmask_inflated_1.0.nii" % subj)
        mask=np.zeros_like(basic_rois).astype(bool)
        mask[basic_rois>0]=1
        mask[voxel_mask_full==0]=0

        nib.save(nib.Nifti1Image(basic_rois,affine),self.mask_root+"subj%02d/func1pt8mm/roi/basic_rois.nii.gz" % subj)
        roi_list=['basic_rois'] + roi_list

        mask=update_mask(mask,roi_list,subj,self.mask_root)
        rois={}
        rois=make_roi_dic(mask,rois,roi_list,subj,self.mask_root,self.nsd_root)


        #masks
        ncsnr_full = load_mask_from_nii(self.beta_folder + "subj%02d/" % subj + self.func_res
                                        + "betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz")


        #voxel_mask = np.nan_to_num(voxel_mask_full).flatten().astype(bool)
        voxel_mask = np.nan_to_num(mask).flatten().astype(bool)
        voxel_ncsnr = ncsnr_full.flatten()[voxel_mask]

        print('full mask length = %d' % len(voxel_mask))
        print('selection length = %d' % np.sum(voxel_mask))

        #load betas
        beta_subj = self.beta_folder + "subj%02d/" % (subj,)+self.func_res+"betas_fithrf_GLMdenoise_RR/"

        voxel_data, filenames = load_betas(folder_name=beta_subj, zscore=zscore,voxel_mask=voxel_mask,
                                           up_to=upto,load_ext=self.load_ext)
        print (voxel_data.shape)

        #training/validation split
        data_size, nv = voxel_data.shape
        trn_stim_data, trn_voxel_data,\
        val_stim_single_trial_data, val_voxel_single_trial_data,\
        val_stim_multi_trial_data, val_voxel_multi_trial_data = \
            data_split(image_feature_fn(image_data), voxel_data, ordering, imagewise=False)
        voxel_data_shape=voxel_data.shape
        del voxel_data

        ##coordinates
        x = np.arange(0,basic_rois.shape[0])
        y = np.arange(0,basic_rois.shape[1])
        z = np.arange(0,basic_rois.shape[2])

        xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')

        x = xx.flatten()[voxel_mask]
        y = yy.flatten()[voxel_mask]
        z = zz.flatten()[voxel_mask]

        # device = torch.device("cuda:0")  # cuda
        # _x = torch.tensor(trn_stim_data[:1]).to(device)  # the input variable.


        out_dict = {'train_stim': trn_stim_data,
                    'val_stim_single': val_stim_single_trial_data,
                    'val_stim_multi': val_stim_multi_trial_data,
                    'train_vox':trn_voxel_data,
                    'val_vox_single':val_voxel_single_trial_data,
                    'val_vox_multi':val_voxel_multi_trial_data,
                    'x':x,'y':y,'z':z,
                    'group': group, 'group_names':group_names,
                    'voxel_ncsnr':voxel_ncsnr,
                    'roi_dic':rois,
                    'mask': mask,
                    'voxel_mask':voxel_mask,
                    'upto': upto,
                    'Nv': voxel_mask.sum(),
                    'voxel_roi_full_path': self.mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz" % subj,
                    'voxel_kast_full_path': self.mask_root + "subj%02d/func1pt8mm/roi/Kastner2015.nii.gz" % (subj),
                    'roi_dic_combined':combine_rois(rois,roi_list[1:],self.nsd_root,subj)}

        self.dic = out_dict
        self.data = namedtuple('dataClass',self.dic)
        self.x=x
        self.y=y
        self.z=z

    def load_nib_viscort(self,file):
        "just a helper function"
        voxel_mask = self.dic['voxel_mask']
        x, y, z = self.dic['x'][voxel_mask], self.dic['y'][voxel_mask], self.dic['z'][voxel_mask]
        return nib.load(file).get_fdata()[x, y, z]

    def load_and_extract_files(self,path,keywords):
        "helper function to load and extract files starting with keyword, also strips extension to return names"
        files = os.listdir(path)
        good_files, names, rois = [],[], []
        for file in files:
            if any(keyword in file for keyword in keywords):
                if file.split('.')[-2]=='nii':
                    good_files.append(file)
                    rois.append(self.load_nib_viscort(path + file))
                    names.append(file.split('.nii.gz')[0])
        return {names[i]:rois[i] for i in range(0,len(names))}


    def specify_rois(self,folder,keywords):
        """
        Here we load the manual localizer rois
        :return: dictionary with rois
        """
        return self.load_and_extract_files(folder, keywords)

    def manual_localizer_rois(self,keywords=['faces','bodies','places','words']):
        """
        Here we load the manual localizer rois
        :return: dictionary with rois
        """
        return self.load_and_extract_files(self.manual_roi_folder,keywords)

    def manual_retinotopy_rois(self, keywords=['prf']):
        """
        Here we load the manually drawn retinotopy rois
        :return:dictionary with rois
        """
        return self.load_and_extract_files(self.manual_roi_folder,keywords)

    def exp_retinotopy_rois(self,keywords=['prf']):
        """
        Here we load the experimental retinotopy rois
        :return: dictionary with rois
        """
        return self.load_and_extract_files(self.exp_roi_folder,keywords)

    def generic_data_loader(self,key_x,key_y,roi=None,batch_size=1,image_size=None):
        """
        A generic data loader
        :param key_x: key for x
        :param key_y: key for y
        :param roi: roi to project y onto
        :param batch_size: batch size
        :return: data loader
        """

        x = torch.from_numpy(self.dic[key_x])
        if image_size is not None:
            resizer=Resize(image_size[-2:])
            x=torch.stack([resizer(xx) for xx in x])
        if roi is None:
            y = torch.from_numpy(self.dic[key_y])
        else:
            y = torch.from_numpy(self.dic[key_y][:, roi])
        return DataLoader(TensorDataset(x,y),batch_size=batch_size)

    def training_data_loader(self,roi=None,image_size=None,batch_size=1):
        return self.generic_data_loader('train_stim','train_vox',roi,batch_size,image_size)

    def validation_data_loader(self,roi=None,image_size=None,batch_size=1):
        return self.generic_data_loader('val_stim_single','val_vox_single',roi,batch_size,image_size)

    # def brain_rois(self):
    #     groups = self.dic['group']
    #     group_names=self.dic['group_names']
    #     voxel_roi=self.dic['voxel_roi'][self.dic['voxel_mask']]
    #     roi_dic={}
    #     for g,group in enumerate(groups):
    #         this_filter=torch.asarray([a in group for a in voxel_roi])
    #         roi_dic[group_names[g]]=this_filter
    #     return roi_dic
    #
    # def brain_and_floc_rois(self):
    #     brain_rois = remove_keys(self.brain_rois(), 'other')
    #     brain_rois.update(greater_than_zero(remove_keys(self.manual_localizer_rois())))
    #     return brain_rois





