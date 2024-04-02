import torch

from utils import *  # layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from dataHandling import dataFetcher
from utils.io_utils import load_config
from boldreams.objectives import *
from models.fmri import prep_encoder, channel_stacker
import os
from attribution import max_activation_stim_fmri
from boldreams import prep_dreamer
from collections import OrderedDict
from torchvision.utils import save_image
from torchvision.transforms import RandomAffine, InterpolationMode
from torch.optim import Adam, SGD


# choose the roi
config_name = sys.argv[1] #"bb-alexnet_upto-16_bbtrain-False_max-channels_100.json"#sys.argv[1]
obj_name = sys.argv[2] #'roi_ref_img_roi_target'#'roi_ref_img'#sys.argv[2]
thresholds = 1800

# get config (this should come from sys.argv)
base = "/home/u2hussai/projects_u/data/"#'/home/uzair/nvme/' #
config = load_config(base + 'configs/' + config_name)

# mkdir to store dream outputs for later visualization
dream_path = base + '/dreams/ref_img/'
if not os.path.exists(dream_path):
    os.makedirs(dream_path)

# load the data
df = dataFetcher(config['base_path'])
df.fetch(upto=config['UPTO'])
sub = 'subj01'
freesurfer_path = base + '/nsd/freesurfer/'

#set up the ref-image dic for each ROI
roi_img_ind={'V1':4870,  'V2':9807,  'V3':7268, 'V3ab':3186,
         'hV4':6565, 'VO':2361,  'LO':2221, 'PHC':2765,
         'IPS':3485, 'MT':4442, 'MST':514,  'floc-bodies':961,
         'floc-places':7969,'floc-words':3409,'prf-eccrois':9909,
             'floc-faces':5382}


# set up the encoder
p_enc = prep_encoder(config, df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = p_enc.get_encoder(device)
enc.load_state_dict(torch.load(model_save_path(config)), strict=False)
rois_dic = df.dic['roi_dic_combined']

for ROI in list(rois_dic.keys()):
    print("working on ROI:", ROI)

    #get the ref_img for the ROI
    ref_img = p_enc.train_data.dataset.tensors[0][roi_img_ind[ROI]].unsqueeze(0)

    if obj_name == 'roi_ref_img':
        rois = OrderedDict({ROI: rois_dic[ROI]})
        p_dreamer = prep_dreamer(enc, rois, p_enc.input_size,ref_img=ref_img.cuda())
        p_dreamer.set_objective('roi_ref_img', roi="roi_" + ROI,
                                ref_img=ref_img.cuda(),gamma=700)
    elif obj_name == 'roi_ref_img_roi_target':
        rois = OrderedDict({ROI: rois_dic[ROI]})
        roi_targets = torch.zeros(rois_dic[ROI].sum())
        p_dreamer = prep_dreamer(enc, rois, p_enc.input_size,ref_img=ref_img.cuda())
        p_dreamer.set_objective('roi_ref_img_roi_target', roi="roi_" + ROI,
                                ref_img=ref_img.cuda(),gamma=700,
                                roi_target=roi_targets.cuda())

    # set the xfm
    if config['backbone_name'] == 'alexnet':
        p_dreamer.set_xfm(xfm=[RandomAffine(5, translate=[0.09, 0.09], fill=0.0)])
    if config['backbone_name'] == 'vgg11':
        p_dreamer.set_xfm(xfm=[RandomAffine(5, translate=[0.09, 0.09], fill=0.0)])
    if (config['backbone_name'] == 'RN50_clip_add_last') or (config['backbone_name'] == 'RN50x4_clip_relu3_last'):
        p_dreamer.set_xfm(xfm=[RandomAffine(2, translate=[0.09, 0.09], scale=[0.4, 0.96], fill=0.0)])
        p_dreamer.set_optim(SGD, lr=4)

    # dream
    test = torch.from_numpy(p_dreamer.dream(thresholds)).moveaxis(-1, 1)

    # plt.subplots()[1].imshow(test[0].moveaxis(0,-1))
    # save the dream and the corresponding fmri
    ref_png_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_ref_img.png'
    dream_png_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_img.png'
    dream_img_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_img.pt'
    dream_fmri_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_fmri.pt'
    dream_fmri = enc(test.float().cuda()).detach().cpu()
    torch.save(test, dream_path + dream_img_name)
    torch.save(dream_fmri, dream_path + dream_fmri_name)
    save_image(test[0], dream_path + dream_png_name)
    save_image(ref_img[0].detach().cpu(), dream_path + ref_png_name)

#setup non-lucent dreamer
# from boldreams.non_lucent import dreamer,roi_ref_img,roi_ref_img_target
#
# ROI='floc-faces'
# drmr = dreamer(enc,lr=3e-3)
# loss=roi_ref_img(rois_dic[ROI],gamma=5,sign=-1)
# #loss=roi_ref_img_target(rois_dic[ROI],torch.zeros(1,rois_dic[ROI].sum()).cuda(),400)
# img_in=p_enc.train_data.dataset.tensors[0][roi_img_ind[ROI]].unsqueeze(0)
# test=drmr.dream(img_in.cuda(),loss,Adam,1200)
# fig,ax=plt.subplots(1,2)
# ax[0].imshow(img_in[0].moveaxis(0,-1))
# ax[1].imshow(test[0].moveaxis(0,-1))

# #test dreamer outside loop
# ROI=('floc-bodies')
# ref_img= p_enc.train_data.dataset.tensors[0][roi_img_ind[ROI]].unsqueeze(0)
# fmri_target= enc(ref_img.cuda())[:,rois_dic[ROI]].detach()
# fmri_target = fmri_target + fmri_target.abs()
# rois = OrderedDict({ROI: rois_dic[ROI]})
# p_dreamer = prep_dreamer(enc, rois, p_enc.input_size,ref_img=ref_img.cuda())
# # p_dreamer.set_objective('roi_ref_img_roi_target',
# #                         roi="roi_" + ROI,
# #                         ref_img=ref_img.cuda(),gamma=100,
# #                         roi_target=fmri_target)
# p_dreamer.set_objective('roi_ref_img',
#                         roi="roi_" + ROI,
#                         ref_img=ref_img.cuda(),gamma=700)
# p_dreamer.set_xfm(xfm=[RandomAffine(1, translate=[0.05, 0.05],scale=[1.0,1.0] ,fill=0.0)])
# p_dreamer.set_optim(Adam,3e-3)
#
# test = torch.from_numpy(p_dreamer.dream(1500)).moveaxis(-1, 1)
#
# fig,ax=plt.subplots(1,2)
# ax[0].imshow(ref_img[0].moveaxis(0,-1))
# ax[1].imshow(test[0].moveaxis(0,-1))

# #
# # #we can figure out the max mean value for an ROI
# # max_mean_roi_gt=p_enc.train_data.dataset.tensors[1][:,rois_dic[ROI]].mean(-1).max()
# # #max_mean_roi_model=max_activation_stim_fmri(enc,rois,p_enc.train_data.dataset.tensors[0],device=enc.device)
# #
# for ROI in list(rois_dic.keys()):
#     print("working on ROI:", ROI)
#     # prepare dreamer
#     # set the objective
#     if obj_name == 'roi_ref_up':
#         rois = OrderedDict({ROI: rois_dic[ROI]})
#         p_dreamer = prep_dreamer(enc, rois, p_enc.input_size)
#         p_dreamer.set_objective('roi', "roi_" + ROI)
#     elif obj_name == 'roi_ref_down':
#         rois = OrderedDict({ROI: rois_dic[ROI], 'not_' + ROI: ~rois_dic[ROI]})
#         roi_targets = torch.asarray([6, 0]).to(enc.device)
#         p_dreamer = prep_dreamer(enc, rois, p_enc.input_size)
#         p_dreamer.set_objective('roi_mean_target', add_prefix_to_keys(rois), roi_targets)
#
#     # set the xfm
#     #if config['backbone_name'] == 'alexnet':
#     p_dreamer.set_xfm(xfm=[RandomAffine(5, translate=[0.01, 0.01], fill=0.0)])
#     #if config['backbone_name'] == 'vgg11':
#     #    p_dreamer.set_xfm(xfm=[RandomAffine(5, translate=[0.09, 0.09], fill=0.0)])
#     #if (config['backbone_name'] == 'RN50_clip_add_last') or (config['backbone_name'] == 'RN50x4_clip_relu3_last'):
#     #    p_dreamer.set_xfm(xfm=[RandomAffine(2, translate=[0.09, 0.09], scale=[0.4, 0.96], fill=0.0)])
#     #    p_dreamer.set_optim(SGD, lr=4)
#
#     # dream
#     test = torch.from_numpy(p_dreamer.dream(thresholds)).moveaxis(-1, 1)
#
#     # plt.subplots()[1].imshow(test[0].moveaxis(0,-1))
#     # save the dream and the corresponding fmri
#     dream_png_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_img.png'
#     dream_img_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_img.pt'
#     dream_fmri_name = config_name.split('.json')[0] + '_roi-' + ROI + '_obj-' + obj_name + '_fmri.pt'
#     dream_fmri = enc(test.float().cuda()).detach().cpu()
#     torch.save(test, dream_path + dream_img_name)
#     torch.save(dream_fmri, dream_path + dream_fmri_name)
#     save_image(test[0], dream_path + dream_png_name)
#
# # dreamer=dream_wrapper(enc,rois)
# # optimizer=lambda params: SGD(params,lr=4)
# # param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=1)
# # jitter_only= [RandomAffine(8,translate=[0.05,0.05],scale=[0.5,0.9], fill=0.0)]
# # obj =roi_mean_target(["roi_"+ROI],torch.asarray([3*max_mean_roi_gt]).cuda()) # roi("roi_"+ROI) #- 1e-1* diversity(
# # # "roi_"+ROI) #+
# # # 1.2*roi_mean_target([
# # # 'roi_v1'],torch.tensor([
# # obj = roi("roi_"+ROI)
# # ##dream
# # _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
# #                 optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(2550,),show_image=False)
# # for img in _[0]:
# #     plt.figure()
# #     plt.imshow(img)
#
#
# import numpy as np
# import torch
# import torchvision.io.image
#
# from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
# import sys
# from dataHandling import dataFetcher
# from utils.io_utils import load_config
# from boldreams.objectives import *
# from boldreams import dream_wrapper,dream_wrapper_ref_img
# from torchvision.transforms import GaussianBlur,Compose,RandomAffine
# from lucent.optvis import render
# from torch.optim import Adam,SGD
# from models.fmri import prep_encoder
# import matplotlib.pyplot as plt
#
#
#
# #backbones
# backbones={'alexnet_on':'alexnet_medium_bbtrain-on_max-channels-100.json',
#            'alexnet_off': 'alexnet_medium_bbtrain-off_max-channels-100.json',
#            'vgg_on': 'bb-vgg11_upto-16_bbtrain-True_max-channels_100.json',
#             'vgg_off':'bb-vgg11_upto-16_bbtrain-False_max-channels_100.json'}
#
# BB='alexnet_off'
# ROI='floc-faces'
#
# #get config
# base='/home/uzair/nvme/'
# config=load_config(base+'/configs/'+backbones[BB])
#
# #get data
# df=dataFetcher(config['base_path'])
# df.fetch(upto=config['UPTO'])
# sub='subj01'
# freesurfer_path = base + '/nsd/freesurfer/'
#
# #get the ROI array
# print(ROI)
# roi_filter=df.dic['roi_dic_combined'][ROI]
# print('Using voxels in ROI:%s'%ROI)
#
# #set up the encoder
# p_enc=prep_encoder(config,df)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# enc=p_enc.get_encoder(device)
# enc.load_state_dict(torch.load(model_save_path(config)),strict=False)
#
# #get images
# imgs=df.training_data_loader(image_size=p_enc.input_size).dataset.tensors[0]
#
# #get top images
# fmri_test = pass_images(imgs, enc, enc.Nv,batch_size=8)
# roi_img_inds=fmri_test[:,roi_filter].mean(-1).argsort(descending=True)
#
# #set up the dreams
# # from boldreams.non_lucent import dreamer
# # from boldreams.non_lucent import activate_roi
# #
# # dream=dreamer(enc,lr=1e-3)
# # img_in=imgs[0]
# # img_out=dream.dream(img_in[0].unsqueeze(0).cuda().float(),activate_roi(roi_filter),Adam,iterations=400)
# #
# # fig,ax=plt.subplots(1,2)
# # ax[0].imshow(img_in[0].moveaxis(0,-1))
# # ax[1].imshow(img_out[0].moveaxis(0,-1))
#
# # #set up the dreaming
# from boldreams.param import ref_image
#
# #input_img
# img_in=imgs[roi_img_inds[0]].unsqueeze(0).cuda().float()
#
# rois={ROI: roi_filter,'not_'+ROI:~roi_filter}
# fmri_target=enc(img_in).detach()
# targets={ROI:-2.5*torch.ones(1,roi_filter.sum()).float().cuda(),'not_'+ROI:fmri_target[:,~roi_filter]}
#
#
# dreamer=dream_wrapper_ref_img(enc,rois,p_enc.input_size,ref_img=ref_img)
# #optimizer=lambda params: SGD(params,lr=0.2)
# optimizer=lambda params: Adam(params,lr=5e-3)
# param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=1)
# param_f=lambda : ref_image(torch.clone(img_in))
# jitter_only= [RandomAffine(0.1,translate=[0.07,0.07])]#,scale=[0.9,1.1], fill=0.0)]
# #obj = -roi("roi_"+ROI) #- 1e-1* diversity("roi_"+ROI) #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([
# # #     # -2]).cuda())
# #obj=roi_targets(targets)
# obj=roi_ref_img("roi_"+ROI,img_in,gamma=1,sign=-1.00)
#
# # #obj=roi_mean_target(["roi_"+ROI],torch.tensor([6.0]).cuda())
# # # obj=roi_mean_target(["roi_"+'FFA-1'],torch.tensor([0.0]).cuda())+roi_mean_target(["roi_"+'FFA-2'],torch.tensor([
# # #     6.0]).cuda()) + roi_mean_target(["roi_"+'OFA'],torch.tensor([0.0]).cuda())
# # # obj=1.0*roi_mean_target(["roi_"+ROI],torch.tensor([5.0]).cuda())#+ 1.5*roi_mean_target(["roi_"+'not-faces'],
# #                                                                  #                     torch.tensor([
# #     #0.0]).cuda())
# ##dream
# _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
#                 optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(1000,),show_image=False)
# for img in _[0]:
#     plt.figure()
#     plt.imshow(img)
#
# plt.figure()
# plt.imshow(img_in[0].detach().cpu().moveaxis(0,-1))
#
# # #get finer rois
# # i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
# # face_rois=roi_from_nii(base+'/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/floc-faces.nii.gz',i,j,k,
# #              freesurfer_path + sub + '/label/floc-faces.mgz.ctab')
# #
# #
# # #set up the dreaming
# # rois=df.dic['roi_dic']
# # not_faces=np.abs(df.dic['roi_dic_combined'][ROI]-1)
# # rois={'floc-faces': torch.ones(np.sum(roi_filter))==1}
# # dreamer=dream_wrapper(enc,rois)
# # #optimizer=lambda params: SGD(params,lr=0.8)
# # optimizer=lambda params: Adam(params,lr=1e-3)
# # param_f = lambda: param.image(p_enc.input_size[-1], fft=True, decorrelate=True,sd=0.01,batch=4)
# # jitter_only= [RandomAffine(1,translate=[0.06,0.06])]#,scale=[0.9,1.1], fill=0.0)]
# # obj = roi("roi_"+ROI) - 1e-5* diversity("roi_"+ROI) #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([
# # #     # -2]).cuda())
# # #obj=roi_mean_target(["roi_"+ROI],torch.tensor([6.0]).cuda())
# # # obj=roi_mean_target(["roi_"+'FFA-1'],torch.tensor([0.0]).cuda())+roi_mean_target(["roi_"+'FFA-2'],torch.tensor([
# # #     6.0]).cuda()) + roi_mean_target(["roi_"+'OFA'],torch.tensor([0.0]).cuda())
# # # obj=1.0*roi_mean_target(["roi_"+ROI],torch.tensor([5.0]).cuda())#+ 1.5*roi_mean_target(["roi_"+'not-faces'],
# #                                                                  #                     torch.tensor([
# #     #0.0]).cuda())
# # ##dream
# # _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
# #                 optimizer=optimizer,fixed_image_size=p_enc.input_size[-1],thresholds=(2350,),show_image=False)
# #
# # #normalize the image
# # fig,ax=plt.subplots(1,len(_[0]))
# # for p,img in enumerate(_[0]):
# #     #img=torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1)
# #     img=img/img.max()
# #     if len(_[0])==1:
# #         ax.imshow(img)
# #     else:
# #         ax[p].imshow(img)
# #
# # ##get fmri and normalize with early signal
# # rois=df.dic['roi_dic_combined']
# # fmri_=enc(img)[0]
# # fmri_=(fmri_-fmri_[np.clip(rois['V1'],0,1)].mean())#/fmri_[np.clip(rois['V1']+rois['V2']+rois[
# #     #'V3'],0,1)].std()
# #
# # #save the dream
# # img_pil=transforms.ToPILImage()(torch.from_numpy(_[-1][0]).moveaxis(-1,0))
# # img_pil.save(base+'/dreams/dream_roi-%s_bb-%s.png'%(ROI,BB))
# # fmri_dream=dreamer(torch.from_numpy(_[-1][0]).unsqueeze(0).cuda().moveaxis(-1,1))
# #
# #
# # #project onto flat brain
# # ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
# # nii=array_to_func1pt8(i,j,k,fmri_.detach().cpu(),ref_nii)
# # surf=func1pt8_to_surfaces(nii,base+'/nsd/',sub,method='linear')[0]
# # flat_hemis=get_flats(freesurfer_path,sub)
# #
# # flat_map_name='flat_map_roi-%s_bb-%s.png'%(ROI,BB)
# # make_flat_map(surf,flat_hemis,base+'/dream_figures/'+flat_map_name,vmin=-3,vmax=4)
# #
# # #make an image with rois
# # ffa_fmri_dream=plt.imread(base+'/dream_figures/'+flat_map_name)
# # ffa_outline=plt.imread(base+'/overlays/'+ '/FFA-outline.png')
# #
# # plt.figure()
# # plt.imshow(ffa_fmri_dream)
# # plt.imshow(ffa_outline,alpha=0.8,cmap='Greys')
# #
# # plt.savefig(base+'/dream_figures/'+flat_map_name)
# #
# # #lets do bar plot
# # rois_for_bar_plot=['V1','V2','V3','floc-faces']
# # fmri_bars=[]
# # for r in rois_for_bar_plot:
# #     fmri_bars.append(fmri_[rois[r]].mean().detach().cpu())
# #
# # fig,ax=plt.subplots()
# # ax.bar(rois_for_bar_plot, fmri_bars)
# #ffa_outline=plt.imread(base+'/overlays/'+ '/FFA-1.png') + plt.imread(base+'/overlays/'+ '/FFA-2.png') + plt.imread(
# # base+'/overlays/'+ '/OFA.png')
# # eba_outline=plt.imread(base+'/overlays/'+ '/FFA-2.png')
# # ofa_outline=plt.imread(base+'/overlays/'+ '/OFA.png')
#
#
#
# # outline=np.abs(np.gradient(ffa_outline)[0]) + np.abs(np.gradient(ffa_outline)[1])
# # outline[outline>0]=1
# # outline[outline==0]=np.nan
# # plt.imshow(eba_outline)
# # plt.imshow(ofa_outline)
# # fmri_nii=array_to_func1pt8(i,j,k,fmri_.detach().cpu(),faces_nii)
# # fmri_surf=func1pt8_to_surfaces(fmri_nii,base+'/nsd/',sub,method='linear')
#
#
