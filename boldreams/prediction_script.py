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


#get config
# config_name=sys.argv[1]
# config=load_config(config_name)

base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-on_max-channels-100.json')

#get data
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])

#lets start at max_percent-1 and then go up to 100
max_ps=[1,5,10,15,20,25,50,75,100]
for max_p in max_ps:
    config_name='-'.join(config_name.split('-')[0:-1]) + '-%d.json'%max_p
    config=load_config(config_name)

    #get encoders
    #initialize encoder
    p_enc=prep_encoder(config,df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc=p_enc.get_encoder(device).float().cuda()

    #load checkpoint
    train_path=model_save_path(config)
    print("load model with path: ", train_path)
    checkpoint=torch.load(train_path)
    enc.load_state_dict(checkpoint,strict=True)

    #make predictions on validation data
    batch_size=1
    val_data=df.validation_data_loader(image_size=p_enc.input_size,batch_size=batch_size)
    fmri_pred=torch.zeros_like(val_data.dataset.tensors[1])
    fmri_gt=torch.zeros_like(val_data.dataset.tensors[1])
    for i,(img,_)in tqdm(enumerate(val_data),total=len(val_data)):
        #print(i*batch_size,(i+1)*batch_size)
        fmri_gt[i*batch_size:(i+1)*batch_size]=_
        fmri_pred[i*batch_size:(i+1)*batch_size]=enc(img.cuda()).detach().cpu()

    #compute the correaltion
    corr=torch.zeros(fmri_pred.shape[1])
    for i in range(0,len(corr)):
        corr[i]=torch.corrcoef(torch.stack([fmri_pred[:,i],fmri_gt[:,i]]))[0,1]

    #have to save correaltion with correct name
    corr_path=train_path.split('.pt')[0]+'_relu_corr.npy'
    np.save(corr_path,corr)

# if config['max_filters']==-1:
#     np.save('corr_max_filters.npy',corr)
# else:
#     corr_large=np.load('corr_max_filters.npy')
#     plt.figure()
#     plt.hist(corr.flatten().numpy(),350,density=1,histtype='step')
#     plt.hist(corr_large,350,density=1,histtype='step')

# config={}
# config['SYSTEM']='local'
# config['UPTO']=6
# config['epochs']=10
# config['max_filters']=-1
# config['train_backbone']='False'
# config['device']=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config['base_path']=get_base_path(config['SYSTEM'])
# df=dataFetcher(config['base_path'])
# df.fetch(upto=config['UPTO'])
#
# #brain_rois=df.brain_and_floc_rois()
# #filter=brain_rois['floc-places']
#
#
# backbones=['alexnet','vgg11','RN50x4_clip_relu3_last']
# corrs={}
# for backbone in backbones:
#     #sys.argv=sysargs(sys.argv,backbone,6,10,-1,'False')
#     #configure
#     config['backbone_name']=backbone
#
#     #initialize encoder
#     p_enc=prep_encoder(config,df)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     enc=p_enc.get_encoder(device)
#
#     #load state_dic
#     train_path=model_save_path(config)
#     checkpoint=torch.load(train_path)
#     enc.load_state_dict(checkpoint,strict=False)
#
#     batch_size=1
#     val_data=df.validation_data_loader(image_size=p_enc.input_size,batch_size=batch_size)
#     fmri_pred=torch.zeros_like(val_data.dataset.tensors[1])
#     fmri_gt=torch.zeros_like(val_data.dataset.tensors[1])
#     for i,(img,_)in tqdm(enumerate(val_data),total=len(val_data)):
#         #print(i*batch_size,(i+1)*batch_size)
#         fmri_gt[i*batch_size:(i+1)*batch_size]=_
#         fmri_pred[i*batch_size:(i+1)*batch_size]=enc(img.cuda()).detach().cpu()
#
#     corr=torch.zeros(fmri_pred.shape[1])
#     #corr = torch.zeros(filter.sum())
#     #fmri_pred=fmri_pred[:,filter]
#     for i in range(0,len(corr)):
#         corr[i]=torch.corrcoef(torch.stack([fmri_pred[:,i],fmri_gt[:,i]]))[0,1]
#     corrs[backbone]=corr
#
# #make volume
# import nibabel as nib
# ref=config['base_path'] + '/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz'
# ref_nii=nib.load(ref)
# corr_vol=np.zeros(ref_nii.shape[0:3])
# filter=df.dic['viscort_roi']
# x=df.dic['x'][filter]
# y=df.dic['y'][filter]
# z=df.dic['z'][filter]
#
# corr_vol[x,y,z]=corrs['alexnet']
#
# nii=nib.Nifti1Image(corr_vol,np.eye(4))
# nib.save(nii,'/home/uzair/Documents/corr.nii.gz')
#
# n=len(corr)
# means=[]
# stds=[]
# confs=[]
# for key,value in corrs.items():
#     mean=value.mean()
#     std=value.std()
#     conf=t.ppf(0.975,df=n-1,scale=std)
#     means.append(value.mean())
#     stds.append(std)
#     confs.append(conf)
#
# labels=['AlexNet','vgg11','RN50x4 (clip)']
# x = np.arange(len(labels))
# #
# # fig, ax = plt.subplots()
# # ax.bar(x, means, yerr=confs, align='center', alpha=0.5, ecolor='black', capsize=10)
# # ax.set_xticks(x)
# # ax.set_xticklabels(labels)
# # ax.set_ylabel('Mean')
# # ax.set_title('Mean and Confidence Interval')
# # ax.yaxis.grid(True)
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 22}
#
# matplotlib.rc('font', **font)
# plt.figure()
# for i,(key,value) in enumerate(corrs.items()):
#     plt.hist(value.detach().cpu().flatten().numpy(),15,density=True,alpha=0.5,
#              histtype='step',linewidth=3,label=labels[i])
# plt.title('Correlation frequency')
# plt.xlabel('Correlation')
# plt.ylabel('Normalized frequency')
# plt.legend()