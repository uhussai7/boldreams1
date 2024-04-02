from matplotlib import gridspec
from torchvision import io


#we need to make publication quality figure for the abstract dreams for wach ROI
#we have activations and also trying to hold other areas at zero
#we have bbtrain-on versus bbtrain-off

import sys

import matplotlib.pyplot as plt

sys.path.append('../../../..')
from utils import *
from torchvision.io import read_image
from dataHandling import dataFetcher
import glob
#make a class for each model will make things a bit easier

base_path=get_base_path('local')
dream_path=base_path+'/dreams/'
ref_img_path=dream_path+'ref_img/'
flat_path=base_path+'/dream_figures/'
plot_path=base_path+'/plots_nsd/gt_vs_model/'

os.makedirs(plot_path,exist_ok=True)

models=['alexnet']#,'vgg11','RN50x4_clip_relu3_last']
bbtrains=['True']#,'False']
max_channels=['1']



#the theme has kinda been bbtrain on vs off. so lets say each figure is for each ROI, coloumns are bbtrain-on-off and
# rows are the different models

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.ioff()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 32,
})

# Create a GridSpec object with relative sizes for rows and columns

# Create subplots using the GridSpec object
overlay_file_names=get_overlay_file_names()
ROIs=overlay_file_names.keys()
labels = [r'a)', r'b)']


fig=plt.figure(figsize=(18,10))
gs=gridspec.GridSpec(1,3,width_ratios=[0.33,1,1])


model=models[0]
bbtrain=bbtrains[0]



for ROI in overlay_file_names.keys():
    for model in models:
        for bbtrain in bbtrains:
            if not (model == 'RN50x4_clip_relu3_last' and bbtrain=='True'):
                for max_channel in max_channels:
                    #load the ref img
                    ref_img='bb-%s_upto-16_bbtrain-%s_max-channels_%s_roi-%s_obj-roi_ref_img_ref_img.png'%(model,
                                                                                                          bbtrain,
                                                                                                           '100',
                                                                                                           ROI)
                    ref_img=read_image(ref_img_path+ref_img)
                    ax=plt.subplot(gs[0])
                    ax.imshow(ref_img.moveaxis(0,-1))
                    ax.set_axis_off()
                    ax.text(-0.1,1.11,'a)',transform=ax.transAxes)

                    #load the model flat maps
                    model_fmri_img='bb-%s_upto-16_bbtrain-%s_max-channels_%s_roi-%s_fmri_model.png'%(model,bbtrain,
                                                                                                     max_channel,
                                                                                                     ROI)
                    model_fmri_img=read_image(flat_path+model_fmri_img)
                    ax=plt.subplot(gs[1])
                    ax.imshow(model_fmri_img.moveaxis(0,-1))
                    ax.set_axis_off()
                    ax.text(-0.05,0.99,'b)',transform=ax.transAxes)

                    #load the gt flat maps
                    gt_fmri_img='ROI-%s_gt_fmri.png'%(ROI)
                    gt_fmri_img=read_image(flat_path+gt_fmri_img)
                    ax=plt.subplot(gs[2])
                    ax.imshow(gt_fmri_img.moveaxis(0,-1))
                    ax.set_axis_off()
                    ax.text(-0.05,0.99,'c)',transform=ax.transAxes)


                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.1, hspace=0)
                    plt.savefig(plot_path + 'gt_vs_fmri_model-%s_bbtrain-%s_max-channels_%s_ROI-%s.png'%(model,
                                                                                                         bbtrain,
                                                                                                         max_channel,
                                                                                                         ROI),
                                bbox_inches='tight',dpi=600)

# for model in models:
#     for ROI in ROIs:
#         if model == 'RN50x4_clip_relu3_last':
#             fig = plt.figure(figsize=(18/2, 13))
#             gs = GridSpec(2, 1, width_ratios=[1], height_ratios=[0.4, 1])
#         else:
#             fig = plt.figure(figsize=(18, 13))
#             gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.4, 1])
#         for b,bbtrain in enumerate(bbtrains):
#             if not (model == 'RN50x4_clip_relu3_last' and bbtrain == 'True'):
#                 dream_img_name='bb-%s_upto-16_bbtrain-%s_max-channels_100_roi-%s_obj-%s_img.png'%(model,bbtrain,ROI,obj)
#                 flat_img_name='dream_model-'+model+'_bbtrain-'+bbtrain+'_obj-'+obj + '_ROI-' + ROI + '.png'
#                 dream_img=read_image(dream_path+dream_img_name)
#                 flat_img=read_image(flat_path+flat_img_name)
#                 print(b)
#                 ax=plt.subplot(gs[0,b])
#                 ax.imshow(dream_img.moveaxis(0,-1))
#                 if model!='RN50x4_clip_relu3_last':
#                     ax.text(-.6,0.96,labels[b],transform=ax.transAxes)
#                 ax.set_axis_off()
#                 ax=plt.subplot(gs[1,b])
#                 ax.imshow(flat_img.moveaxis(0,-1))
#                 ax.set_axis_off()
#         plt.tight_layout()
#         plt.subplots_adjust(hspace=0.005)  # Change the hspace parameter here
#         plt.savefig(plot_path+'dream_figure_model-%s_ROI-%s_obj-%s.png'%(model,ROI,obj),dpi=600)