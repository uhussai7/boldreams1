from dataHandling import dataFetcher
import sys
import torch
from utils import *  # layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from torchvision.models.feature_extraction import get_graph_node_names
from models.fmri import prep_encoder
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from utils import trainer

# get config
# config = load_config('/home/u2hussai/projects_u/data/configs/bb-alexnet_upto-16_bbtrain-True_max-channels_100.json' #load_config(sys.argv[1])#
config = load_config(sys.argv[1])  #
df = dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])

# get the ROI array
try:
    ROI = str(sys.argv[2])
    print(ROI)
    roi_filter = df.dic['roi_dic_combined'][ROI]
    print('Using voxels in ROI:%s' % ROI)
except:
    print('Using all voxels')
    roi_filter = None
    ROI = None

# initialize encoder
p_enc = prep_encoder(config, df, roi=roi_filter)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = p_enc.get_encoder(device).float().cuda()

# train
# decide which parameters are to be updated
if bool(config['train_backbone']) == True:
    print('Going to train backbone')
    params_to_pass = enc.parameters()
else:
    params_to_pass = [{'params': enc.rfs.parameters()},
                      {'params': enc.w_2d},
                      # {'params':enc.w_1d},
                      {'params': enc.b}]

optimizer = Adam(params=params_to_pass, lr=0.0001)

trnr = trainer(enc, p_enc.train_data,
               optimizer, MSELoss(),
               max_epochs=config['epochs'],
               scheduler=MultiStepLR(optimizer, milestones=[2, 3, 4, 5, 6, 7], gamma=0.8))
enc = trnr.fit()

# decide what needs to be saved
if bool(config['train_backbone']) == True:
    state_dict = enc.state_dict()
else:
    params_to_save = ['w_2d', 'b'] + [a for a in enc.state_dict().keys() if a.split('.')[0] == 'rfs']
    state_dict = {param_name: enc.state_dict()[param_name] for param_name in params_to_save}

print('Saving model to ' + model_save_path(config, ROI))
torch.save(state_dict, model_save_path(config, ROI))
