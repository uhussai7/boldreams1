from utils.io_utils import load_config,model_save_path
from dataHandling import dataFetcher
from attribution import filter_attribution
import os

base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
df=dataFetcher(config['base_path'])

UPTO_=1 #warning: changing upto for speed, sample should be large enough
df.fetch(upto=UPTO_)

config_=config.copy()
config_['UPTO']=UPTO_
filter_rank_file=model_save_path(config_).split('.pt',1)[0] + '_ranks.pt'

#roi dic
roi_dic={}
for roi in df.dic['group_names'][:-1]:
    roi_dic[roi]=df.dic['roi_dic_combined'][roi]

att=filter_attribution(config,df)
ranks=att.mean_rank_of_filters_over_rois(roi_dic,filter_rank_file)#imgs_inds=[0])