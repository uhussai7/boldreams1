import numpy as np
import matplotlib.pyplot as plt
import torch
from dataHandling import dataFetcher
from scipy.stats import ttest_ind
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer


base='/home/uzair/nvme/'
# df=dataFetcher(base)
# df.fetch(upto=1)

def get_corr(backbone_name,train_or_not,Nf):
    corr_name= '/trainedEncoders/corrs/backbone_name-%s_UPTO-16_epochs-10_train_backbone-%s_max_filters' \
            '-False_max_percent-%s_trained_corr.pt' %(backbone_name,train_or_not,Nf)
    return torch.load(base+corr_name).cpu().numpy()




base_path=get_base_path('local')
df=dataFetcher(base_path)
df.fetch(upto=1)
noise=df.dic['voxel_ncsnr']


#choose the roi
config_name='bb-alexnet_upto-16_bbtrain-False_max-channels_100.json'#sys.argv[1]
#ROI=sys.argv[2] #lets do all Rois
obj_name='roi_spec'#sys.argv[2]

#get config (this should come from sys.argv)
base='/home/uzair/nvme/'
config=load_config(base+'configs/' + config_name)
freesurfer_path = base + '/nsd/freesurfer/'

#overlay path
overlay_path=base_path+'/overlays/'
ROI='floc-faces'
flat_maps=DreamFlatmaps(base_path,df.dic)
corr_diff=get_corr('alexnet','False',25)-get_corr('vgg11','True',1)
flat_maps.make_signal_flatmap(corr_diff, get_overlay_file_names()[ROI]+get_overlay_file_names()['V1'],
                              'fine_tuning_voxels.png',
                              vmin=-1,vmax=1)
#need to add a title





# #plt params
# #okay make the plot
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
#     "font.size": 22,
# })
#
# fig,ax=plt.subplots(figsize=(9,9))
# ax.hexbin(corr_diff,noise,cmap='inferno',bins=30,mincnt=1)
# ax.set_xlim([-0.7,0.7])
# plt.show()