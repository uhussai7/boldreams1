from dataHandling import dataFetcher
import sys
import torch
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from torch.nn import MSELoss
from torch.optim import Adam,SGD
from torchvision.models.feature_extraction import get_graph_node_names
from models.fmri import prep_encoder
from torch.optim.lr_scheduler import MultiStepLR,StepLR



#get config
base=get_base_path('cedar')
config_bb_off = load_config(base+'/configs/alexnet_medium_bbtrain-off_max-channels-100.json')
#config=load_config('./configs/alexnet_small_bbtrain-off.json')

df=dataFetcher(config_bb_off['base_path'])
df.fetch(upto=config_bb_off['UPTO'])

#initialize encoder
p_enc=prep_encoder(config_bb_off,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device).float().cuda()

#load the state_dic
enc=load_checkpoint(config_bb_off,enc)


#we will train all the parameters now
params_to_pass = enc.parameters()

from utils import trainer
optimizer=Adam(params=params_to_pass,lr=0.0001)

trnr=trainer(enc,p_enc.train_data,
             optimizer,MSELoss(),
             max_epochs=config_bb_off['epochs'],
             scheduler= MultiStepLR(optimizer, milestones=[2,3,4,5,6,7], gamma=0.8))
enc=trnr.fit()
state_dict=enc.state_dict()#{param_name: enc.state_dict()[param_name] for param_name in params_to_save}

config_bb_off['off2on']='True'#
print('Saving model to '+model_save_path(config_bb_off))
torch.save(state_dict,model_save_path(config_bb_off))



