import torch

from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
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

#choose the roi
config_name='bb-vgg_upto-16_bbtrain-True_max-channels_1.json'#sys.argv[1]
#ROI=sys.argv[2] #lets do all Rois
obj_name='roi'#sys.argv[2]
thresholds=2600

#get config (this should come from sys.argv)
base='/home/uzair/nvme/'#'/home/u2hussai/projects_u/data/'
config=load_config(base+'configs/' + config_name)

#mkdir to store dream outputs for later visualization
dream_path=base+'/dreams/'
if not os.path.exists(dream_path):
    os.makedirs(dream_path)

#load the data
df=dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])
sub='subj01'
freesurfer_path = base + '/nsd/freesurfer/'

#set up the encoder
p_enc=prep_encoder(config,df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc=p_enc.get_encoder(device)
enc.load_state_dict(torch.load(model_save_path(config)),strict=False)
rois_dic=df.dic['roi_dic_combined']

    

#
# #we can figure out the max mean value for an ROI
# max_mean_roi_gt=p_enc.train_data.dataset.tensors[1][:,rois_dic[ROI]].mean(-1).max()
# #max_mean_roi_model=max_activation_stim_fmri(enc,rois,p_enc.train_data.dataset.tensors[0],device=enc.device)
#
for ROI in list(rois_dic.keys()):
    print("working on ROI:",ROI)
    #prepare dreamer
    #set the objective
    if obj_name=='roi':
        rois = OrderedDict({ROI: rois_dic[ROI]})
        p_dreamer = prep_dreamer(enc, rois, p_enc.input_size)
        p_dreamer.set_objective('roi',"roi_"+ROI)
    elif obj_name=='roi_spec':
        rois = OrderedDict({ROI: rois_dic[ROI],'not_'+ROI: ~rois_dic[ROI]})
        roi_targets=torch.asarray([6,0]).to(enc.device)
        p_dreamer = prep_dreamer(enc, rois, p_enc.input_size)
        p_dreamer.set_objective('roi_mean_target',add_prefix_to_keys(rois),roi_targets)

    #set the xfm
    if config['backbone_name']=='alexnet':
        p_dreamer.set_xfm(xfm=[RandomAffine(5,translate=[0.09,0.09], fill=0.0)])
    if config['backbone_name']=='vgg11':
        p_dreamer.set_xfm(xfm=[RandomAffine(5,translate=[0.09,0.09], fill=0.0)])
    if (config['backbone_name']=='RN50_clip_add_last') or (config['backbone_name']=='RN50x4_clip_relu3_last'):
        p_dreamer.set_xfm(xfm=[RandomAffine(2,translate=[0.09,0.09], scale=[0.4,0.96], fill=0.0)])
        p_dreamer.set_optim(SGD,lr=4)

        
    #dream
    test=torch.from_numpy(p_dreamer.dream(thresholds)).moveaxis(-1,1)

    # plt.subplots()[1].imshow(test[0].moveaxis(0,-1))
    #save the dream and the corresponding fmri
    dream_png_name=config_name.split('.json')[0]+'_roi-'+ROI+'_obj-'+obj_name+'_img.png'
    dream_img_name=config_name.split('.json')[0]+'_roi-'+ROI+'_obj-'+obj_name+'_img.pt'
    dream_fmri_name=config_name.split('.json')[0]+'_roi-'+ROI+'_obj-'+obj_name+'_fmri.pt'
    dream_fmri=enc(test.float().cuda()).detach().cpu()
    torch.save(test,dream_path+dream_img_name)
    torch.save(dream_fmri,dream_path+dream_fmri_name)
    save_image(test[0],dream_path+dream_png_name)







# dreamer=dream_wrapper(enc,rois)
# optimizer=lambda params: SGD(params,lr=4)
# param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=1)
# jitter_only= [RandomAffine(8,translate=[0.05,0.05],scale=[0.5,0.9], fill=0.0)]
# obj =roi_mean_target(["roi_"+ROI],torch.asarray([3*max_mean_roi_gt]).cuda()) # roi("roi_"+ROI) #- 1e-1* diversity(
# # "roi_"+ROI) #+
# # 1.2*roi_mean_target([
# # 'roi_v1'],torch.tensor([
# obj = roi("roi_"+ROI)
# ##dream
# _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
#                 optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(2550,),show_image=False)
# for img in _[0]:
#     plt.figure()
#     plt.imshow(img)
