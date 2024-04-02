import torch
from utils import stims
import numpy as np
import torchvision
import numpy as np
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
import sys
from dataHandling import dataFetcher
from utils.io_utils import load_config
from lucent.optvis import render
from boldreams.objectives import *
from models.fmri import prep_encoder
from boldreams import dream_wrapper
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render
from torch.optim import Adam,SGD
from models.fmri import prep_encoder
import matplotlib.pyplot as plt
from nibabel.freesurfer.io import *
from utils import closestDivisors,pass_images
from attribution import integrated_gradient_terms
from models.fmri import prep_encoder,channel_stacker
from attribution import integrated_gradient,integrated_gradient_vox


def load_data(config):
    df=dataFetcher(config['base_path'])
    df.fetch(upto=config['UPTO'])
    return df

def load_model(config):
    p_enc=prep_encoder(config,df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc=p_enc.get_encoder(device)
    print('loading from:',model_save_path(config))
    enc.load_state_dict(torch.load(model_save_path(config)),strict=False)
    return p_enc,enc

def compute_corr(p_enc,enc,config,df,batch_size=2):
    val_data = df.validation_data_loader(image_size=p_enc.input_size)
    imgs = val_data.dataset.tensors[0]
    fmri_gt=val_data.dataset.tensors[1]
    Nv=p_enc.Nv
    fmri_model=torch.zeros(fmri_gt.shape[0],Nv)
    print('passing images through model')
    for i in range(0,imgs.shape[0],batch_size):
        fmri_model[i:i+batch_size]=enc(imgs[i:i+batch_size].float().cuda()).detach().cpu()
    corrt=torch.zeros(Nv)
    print('comuting correlation')
    for i in range(0,Nv):
        corrt[i]=torch.corrcoef(torch.stack([fmri_model[:,i],fmri_gt[:,i]]))[0,1]
    corr_save_path=model_save_path(config).split('.pt')[0] + '_corr.pt'
    print('save corr in:',corr_save_path)
    torch.save(corrt,corr_save_path)
    return corrt

#load any config and it will vary the Nf for it
base='/home/u2hussai/projects_u/data/'#'/home/uzair/nvme/'
backbone_name=str(sys.argv[1])
train_or_not=str(sys.argv[2])
#first load the data
config_name=base+'/configs/'+ \
    'bb-%s_upto-16_bbtrain-%s_max-channels_%d.json'%(backbone_name,train_or_not,100)
config=load_config(config_name)
df=load_data(config)

Nfs=[1,5,10,15,20,25,50,75,100]
for Nf in Nfs:
    config_name=base+'/configs/'+ \
    'bb-%s_upto-16_bbtrain-%s_max-channels_%d.json'%(backbone_name,train_or_not,Nf)
    config=load_config(config_name)
    p_enc,enc=load_model(config)
    this_corr=compute_corr(p_enc,enc,config,df)





