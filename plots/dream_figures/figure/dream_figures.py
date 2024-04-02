from torchvision import io


#we need to make publication quality figure for the abstract dreams for wach ROI
#we have activations and also trying to hold other areas at zero
#we have bbtrain-on versus bbtrain-off

import sys
import numpy as np

import matplotlib.pyplot as plt

sys.path.append('../../../..')
from utils import *
from torchvision.io import read_image
from dataHandling import dataFetcher
import glob
#make a class for each model will make things a bit easier

base_path=get_base_path('local')
dream_path=base_path+'/dreams/'
flat_path=base_path+'/dream_figures/'
plot_path=base_path+'/plots_nsd/dreams/'

os.makedirs(plot_path,exist_ok=True)

models=['alexnet','vgg11','RN50x4_clip_relu3_last']
bbtrains=['False','True']
max_channels=['100']
objs=['roi','roi_spec']



#the theme has kinda been bbtrain on vs off. so lets say each figure is for each ROI, coloumns are bbtrain-on-off and
# rows are the different models

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.ioff()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 19,
})

# Create a GridSpec object with relative sizes for rows and columns

# Create subplots using the GridSpec object
overlay_file_names=get_overlay_file_names()
ROIs=overlay_file_names.keys()
labels = [r'a)', r'b)', r'c)', r'd)', r'e)']
st=np.asarray([1,3,0,2,4])
en=st+2
row={'alexnet':0,'vgg11':2,'RN50x4_clip_relu3_last':2}
backbone_names=[r'\noindent AlexNet \\ Type I',
                r'\noindent AlexNet \\ Type II',
                r'\noindent VGG \\ Type I',
                r'\noindent VGG \\ Type II',
                #r'\noindent CLIP \\ RN50 \\Type I',
                r'\noindent CLIP \\ RN50x4 \\Type I']
for ROI in ROIs:
    for obj in objs:
        fig = plt.figure(figsize=(21/2, 23/2))
        gs = GridSpec(4, 6, height_ratios=[0.4, 1,0.4,1])
        ll=0
        for model in models:
                for b,bbtrain in enumerate(bbtrains):
                    if not (model == 'RN50x4_clip_relu3_last' and bbtrain == 'True'):
                        dream_img_name='bb-%s_upto-16_bbtrain-%s_max-channels_100_roi-%s_obj-%s_img.png'%(model,bbtrain,ROI,obj)
                        flat_img_name='dream_model-'+model+'_bbtrain-'+bbtrain+'_obj-'+obj + '_ROI-' + ROI + '.png'
                        dream_img=read_image(dream_path+dream_img_name)
                        flat_img=read_image(flat_path+flat_img_name)
                        #print(b)
                        ax=plt.subplot(gs[row[model],st[ll]:en[ll]])
                        ax.imshow(dream_img.moveaxis(0,-1))
                        #if model!='RN50x4_clip_relu3_last':
                        ax.text(-.6,0.96,labels[ll],transform=ax.transAxes)
                        ax.text(-0.54,0.55,backbone_names[ll],transform=ax.transAxes,size=15)
                        ax.set_axis_off()
                        ax=plt.subplot(gs[row[model]+1,st[ll]:en[ll]])
                        ax.imshow(flat_img.moveaxis(0,-1))
                        ax.set_axis_off()
                        print(ll)
                        ll+=1
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.005,wspace=0.05)  # Change the hspace parameter here
        plt.savefig(plot_path+'dream_figure_ROI-%s_obj-%s.png'%(ROI,obj),dpi=450,bbox_inches='tight')

        #
# overlay_file_names=get_overlay_file_names()
# ROIs=overlay_file_names.keys()
# labels = [r'a)', r'b)']
#
# for model in models:
#     for ROI in ROIs:
#         for obj in objs:
#             if model == 'RN50x4_clip_relu3_last':
#                 fig = plt.figure(figsize=(18/2, 13))
#                 gs = GridSpec(2, 1, width_ratios=[1], height_ratios=[0.4, 1])
#
#             else:
#                 fig = plt.figure(figsize=(18, 13))
#                 gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[0.4, 1])
#             for b,bbtrain in enumerate(bbtrains):
#                 if not (model == 'RN50x4_clip_relu3_last' and bbtrain == 'True'):
#                     dream_img_name='bb-%s_upto-16_bbtrain-%s_max-channels_100_roi-%s_obj-%s_img.png'%(model,bbtrain,ROI,obj)
#                     flat_img_name='dream_model-'+model+'_bbtrain-'+bbtrain+'_obj-'+obj + '_ROI-' + ROI + '.png'
#                     dream_img=read_image(dream_path+dream_img_name)
#                     flat_img=read_image(flat_path+flat_img_name)
#                     print(b)
#                     ax=plt.subplot(gs[0,b])
#                     ax.imshow(dream_img.moveaxis(0,-1))
#                     if model!='RN50x4_clip_relu3_last':
#                         ax.text(-.6,0.96,labels[b],transform=ax.transAxes)
#                     ax.set_axis_off()
#                     ax=plt.subplot(gs[1,b])
#                     ax.imshow(flat_img.moveaxis(0,-1))
#                     ax.set_axis_off()
#             plt.tight_layout()
#             plt.subplots_adjust(hspace=0.005)  # Change the hspace parameter here
#             plt.savefig(plot_path+'dream_figure_model-%s_ROI-%s_obj-%s.png'%(model,ROI,obj),dpi=600,bbox_inches='tight')