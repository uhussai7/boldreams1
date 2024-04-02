import clip
import numpy as np
from dataHandling import dataFetcher
import sys
import torch
from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from torch.nn import MSELoss
from torch.optim import Adam,SGD
from torchvision.models.feature_extraction import get_graph_node_names
from models.fmri import prep_encoder
import matplotlib.pyplot as plt
from scipy.stats import t
import matplotlib
from attribution import *
import matplotlib.gridspec as gridspec


def get_ground_truth_fmri(df,fmri,fmri_top_inds,roi_list,type):
    if type=='training':
        fmri['gt']=max_activation_stim_fmri_gt(roi_list,df.dic['train_vox'])
    elif type == 'testing':
        fmri['gt'] = max_activation_stim_fmri_gt(roi_list, df.dic['val_vox_single'])
        fmri_top_inds['gt']=max_activation_stim_inds(fmri['gt'])
    return fmri,fmri_top_inds

def get_backbone_fmri(backbone,config,fmri,fmri_top_inds,type):
    config['backbone_name'] = backbone
    p_enc = prep_encoder(config, df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = p_enc.get_encoder(device)
    # load state_dic
    train_path = model_save_path(config)
    checkpoint = torch.load(train_path)
    enc.load_state_dict(checkpoint, strict=False)
    if type=='training':
        imgs=p_enc.train_data.dataset.tensors[0]
    elif type =='testing':
        val_data = df.validation_data_loader(image_size=p_enc.input_size)
        imgs = val_data.dataset.tensors[0]
    fmri[backbone] = max_activation_stim_fmri(enc, rois, imgs)
    fmri_top_inds[backbone] = max_activation_stim_inds(fmri[backbone])
    return fmri,fmri_top_inds,p_enc.input_size

def get_train_backbone_fmri(backbone,config,fmri,fmri_top_inds,type): #make backbone an informative key
    #config['backbone_name'] = backbone
    p_enc = prep_encoder(config, df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = p_enc.get_encoder(device)
    # load state_dic
    train_path = model_save_path(config)
    checkpoint = torch.load(train_path)
    enc.load_state_dict(checkpoint, strict=False)
    if type=='training':
        imgs=p_enc.train_data.dataset.tensors[0]
    elif type =='testing':
        val_data = df.validation_data_loader(image_size=p_enc.input_size)
        imgs = val_data.dataset.tensors[0]
    fmri[backbone] = max_activation_stim_fmri(enc, rois, imgs)
    fmri_top_inds[backbone] = max_activation_stim_inds(fmri[backbone])
    return fmri,fmri_top_inds,p_enc.input_size

def avg_rank_agreement(fmri_top_inds,r,cut=10):
    backbones_=list(fmri_top_inds.keys())
    rank_overlap = torch.zeros((len(backbones_), len(backbones_)))
    for i, b1 in enumerate(backbones_):
        ranks = fmri_top_inds[b1][r][:cut]
        for j, b2 in enumerate(backbones_):
            rank_overlap[i, j] = torch.asarray([torch.where(fmri_top_inds[b2][r] == a)[0].numpy()[0] - a_i for a_i, a in
                                                enumerate(ranks)]).float().abs().mean()
    return rank_overlap

def accuracy(fmri,r):
    backbones=list(fmri.keys())
    corr={}
    for b in backbones:
        nv=fmri[b][r].shape[1]
        corrt=torch.zeros(nv)
        for i in range(0,nv):
            corrt[i]=torch.corrcoef(torch.stack([fmri[b][r][:,i],fmri['gt'][r][:,i]]))[0,1]
        corr[b]=corrt
    return corr

def get_imgs(df,type,input_size):
    if type=='training':
        return df.training_data_loader(image_size=input_size).dataset.tensors[0]
    if type=='texting':
        return df.validation_data_loader(image_size=input_size).dataset.tensors[0]

def remove_repeats(n_return,imgs_ordered):
    imgs_out=[]
    imgs_out.append(imgs_ordered[0])
    for img in imgs_ordered:
        if (img-imgs_out[-1]).abs().mean() > 1e-6:
            imgs_out.append(img)
    return torch.stack(imgs_out)[:n_return]

#do the config and get data

config={}
config['SYSTEM']='local'
config['UPTO']=16
config['epochs']=10
config['max_filters']=-1
config['train_backbone']='False'
config['device']=torch.device("cuda" if torch.cuda.is_available() else "cpu")
config['base_path']=get_base_path(config['SYSTEM'])
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])
rois=df.dic['roi_dic_combined']


#backbones=['alexnet','vgg11','RN50x4_clip_relu3_last']
backbones=['alexnet']#,'vgg11','RN50x4_clip_relu3_last']
type='testing'
fmri={}
fmri_top_inds={}
img_size={}
fmri,fmri_top_inds=get_ground_truth_fmri(df,fmri,fmri_top_inds,df.dic['roi_dic_combined'],type)
#for backbone in backbones:
config['backbone_name']='alexnet'
config_train_bb_true=config.copy()
config_train_bb_true['train_backbone']=True
useful_keys=['no','yes']
for kk,config_ in enumerate([config,config_train_bb_true]):
    fmri,fmri_top_inds,img_size[useful_keys[kk]]=get_train_backbone_fmri(useful_keys[kk],config_,fmri,fmri_top_inds,type)

#for each backbone and each roi plot top 4 stimuli
plots_path='/home/uzair/Documents/'
#rois=['floc-faces','floc-places','floc-bodies']
rois=['floc-faces']#,'floc-places','floc-bodies']
top_n=4
s=int(np.sqrt(top_n))
fig = plt.figure(figsize=(15,15))
#outer=gridspec.GridSpec(len(backbones),len(rois),wspace=0.05,hspace=0.05)
outer=gridspec.GridSpec(len(useful_keys),len(rois),wspace=0.05,hspace=0.05)
#for b_i,b in enumerate(backbones):
for b_i,b in enumerate(useful_keys):
    for r_i,r in enumerate(rois):
        innner=gridspec.GridSpecFromSubplotSpec(s,s,subplot_spec=outer[b_i,r_i],wspace=0,hspace=0)
        inds=fmri_top_inds[b][r][:int(top_n*3)]
        print(b)
        print(inds)
        print('\n')
        imgs=get_imgs(df,type,img_size[b])[inds]
        imgs=remove_repeats(top_n,imgs)
        k=0
        for i in range(0,s):
            for j in range(0,s):
                ax=plt.Subplot(fig,innner[i,j])
                ax.imshow(imgs[k].moveaxis(0,-1).detach().cpu().numpy())
                ax.axis('off')
                fig.add_subplot(ax)
                k+=1

corr=accuracy(fmri,'floc-faces')