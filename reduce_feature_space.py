import clip
from dataHandling import dataFetcher
import sys
import torch
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from torch.nn import MSELoss
from torch.optim import Adam,SGD
from torchvision.models.feature_extraction import get_graph_node_names
from models.fmri import prep_encoder
#from boldreams import dream_wrapper
import matplotlib.pyplot as plt
#from boldreams.objectives import *
#from boldreams.param import ref_image
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
#from lucent.optvis import render
from torchvision.utils import save_image
import numpy as np

#from utils.surface import *
#from mayavi import mlab
from dataHandling import dataFetcher
import nibabel as nib
from utils.roi_utils import anat_combine_rois,vol_for_flat_maps
from utils.io_utils import load_config
#from attribution.feature import *
from torch.optim.lr_scheduler import MultiStepLR,StepLR

#get config
config=load_config(sys.argv[1])
print(config['base_path'])
#get data
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])

#get encoders
#initialize encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device).float().cuda()

#train
if bool(config['train_backbone']) == True:
    print('Going to train backbone')
    params_to_pass= enc.parameters()
else:
    params_to_pass=[{'params':enc.rfs.parameters()},
                {'params':enc.w_2d},
                #{'params':enc.w_1d},
                {'params':enc.b}]
from utils import trainer
optimizer=Adam(params=params_to_pass,lr=0.0001)

trnr=trainer(enc,p_enc.train_data,
             optimizer,MSELoss(),
             max_epochs=config['epochs'],
             scheduler= MultiStepLR(optimizer, milestones=[2,3,4,5,6,7], gamma=0.8))
enc=trnr.fit()
if bool(config['train_backbone'])==True:
    state_dict=enc.state_dict()
else:
    params_to_save=['w_2d','b'] + [a for a in enc.state_dict().keys() if a.split('.')[0]=='rfs']
    state_dict = {param_name: enc.state_dict()[param_name] for param_name in params_to_save}

print('Saving model to '+model_save_path(config))
torch.save(state_dict,model_save_path(config))

