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
dream_path=base_path+'/dreams/ref_img/'
flat_path=base_path+'/dream_figures/'
plot_path=base_path+'/plots_nsd/dreams/'

os.makedirs(plot_path,exist_ok=True)

models=['alexnet','vgg11','RN50x4_clip_relu3_last']
bbtrains=['False','True']
max_channels=['100']
objs=['roi_ref_img','roi_ref_img_roi_target']


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

for model in models:
   for ROI in ROIs:
       for obj in objs:
            print(model,ROI,obj)
            if model == 'RN50x4_clip_relu3_last':
                fig = plt.figure(figsize=(13, 13))
                gs = GridSpec(2, 2, height_ratios=[0.4, 1])
                labels={'False':[r'b)',r'c)']}
                ax = plt.subplot(gs[0, 0:1])
            else:
                fig = plt.figure(figsize=(18, 13))
                gs = GridSpec(2, 6, height_ratios=[0.4, 1])
                labels={'False':[r'b)',r'd)'], 'True':[r'c)',r'e)']}
                ax = plt.subplot(gs[0, 0:2])
            try:
                ref_img = 'bb-%s_upto-16_bbtrain-%s_max-channels_100_roi-%s_obj-roi_ref_img_ref_img.png' % (model, 'True', ROI)
                ref_img = read_image(dream_path + ref_img)
            except:
                ref_img = 'bb-%s_upto-16_bbtrain-%s_max-channels_100_roi-%s_obj-roi_ref_img_ref_img.png' % (model, 'False', ROI)
                ref_img = read_image(dream_path + ref_img)
            ax.set_axis_off()
            ax.imshow(ref_img.moveaxis(0,-1))
            ax.text(-0.2,0.96,r'a)',transform=ax.transAxes)
            for b,bbtrain in enumerate(bbtrains):
                if not (model == 'RN50x4_clip_relu3_last' and bbtrain == 'True'):
                    label=labels[bbtrain]
                    dream_img_name='bb-%s_upto-16_bbtrain-%s_max-channels_100_roi-%s_obj-%s_img.png'%(model,bbtrain,ROI,obj)
                    flat_img_name='dream_model-%s_bbtrain-%s_ROI-%s_obj-%s.png'%(model,bbtrain,ROI,obj)
                    dream_img=read_image(dream_path+dream_img_name)
                    flat_img=read_image(flat_path+flat_img_name)
                    print(b)
                    if model=='RN50x4_clip_relu3_last':
                        ax = plt.subplot(gs[0,b+1:(b+2)])
                    else:
                        ax=plt.subplot(gs[0,(2*b+2):(2*b+4)])
                    ax.imshow(dream_img.moveaxis(0,-1))
                    #if model!='RN50x4_clip_relu3_last':
                    ax.text(-.2,0.96,label[0],transform=ax.transAxes)
                    ax.set_axis_off()
                    ax=plt.subplot(gs[1,3*b:(3*b+3)])
                    ax.imshow(flat_img.moveaxis(0,-1))
                    ax.text(-.05,0.92,label[1],transform=ax.transAxes)
                    ax.set_axis_off()
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.005)  # Change the hspace parameter here
            plt.savefig(plot_path+'dream_figure_model-%s_ROI-%s_obj-%s_ref_img.png'%(model,ROI,obj),dpi=600,bbox_inches='tight')