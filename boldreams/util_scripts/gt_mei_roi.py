import torch

from dataHandling import dataFetcher
from utils.io_utils import load_config
from boldreams.objectives import *
from models.fmri import prep_encoder,channel_stacker
import os
from attribution import max_activation_stim_fmri
from boldreams import prep_dreamer
from collections import OrderedDict
from torchvision.utils import save_image
from torchvision.transforms import RandomAffine
from torch.optim import Adam,SGD
import torchvision
from tqdm import tqdm

#get config (this should come from sys.argv)
base_path='/home/uzair/nvme/'


#load the data
df=dataFetcher(base_path)
df.fetch(upto=16)
sub='subj01'

#make the dir
os.makedirs(base_path+'/meis/max_gt_mei/',exist_ok=True)

#we need to store the rank and also the index in main dataset.
#since images need to be sorted we will store with filename rank and another textfile with image_index
N_max=100 #we will only sae top N_max images
rois_dic=df.dic['roi_dic_combined']
fmri_gt=torch.from_numpy(df.dic['train_vox'])
fmri_stim=torch.from_numpy(df.dic['train_stim'])
for roi_name,roi_filter in rois_dic.items():
    print('Working on roi',roi_name)
    #make a different folder for each roi
    img_path=base_path+'/meis/max_gt_mei/'+roi_name + '/'
    os.makedirs(img_path,exist_ok=True)
    roi_inds=fmri_gt[:,roi_filter].mean(-1).argsort(descending=True)
    rank_ind_file_name = img_path + 'rank_ind_.txt'
    with open(rank_ind_file_name,'w') as file:
        for rank in tqdm(range(N_max)):
            roi_ind=roi_inds[rank]
            torchvision.utils.save_image((fmri_stim[roi_ind]),img_path+'/%d.png'%rank)
            file.write(str('rank=%d ind=%d\n'%(rank,roi_ind)))

