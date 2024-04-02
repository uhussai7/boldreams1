import sys

import matplotlib.pyplot as plt

sys.path.append('../../..')
from utils import *
from dataHandling import dataFetcher
import glob
#make a class for each model will make things a bit easier

base_path=get_base_path('local')
df=dataFetcher(base_path)
df.fetch(upto=1)

#we are loading dreams of 5 models
dreams_path=base_path+'/dreams/'
overlay_path=base_path+'/overlays/'

models=['alexnet','vgg11','RN50x4_clip_relu3_last']
bbtrains=['True','False']
max_channels=['100']
objs=['roi','roi_spec']

#earier to just look at alexnet and decide which ROIs go together

overlay_file_names=get_overlay_file_names()

dream_flats=DreamFlatmaps(base_path,df.dic)
for ROI in overlay_file_names.keys():
    for obj in objs:
        for model in models:
            for max_channel in max_channels:
                for bbtrain in bbtrains:
                    if not (model == 'RN50x4_clip_relu3_last' and bbtrain=='True'):
                        file_name=('dream_model-'+model+'_bbtrain-'+bbtrain+'_max_channels-'+max_channel+'_obj-'+obj +
                                   '_ROI-' + ROI+ '.png')
                        print(file_name)
                        dream_flats.make_dream_flatmap(model,bbtrain,ROI,obj,overlay_file_names[ROI],
                                                       out_filename=file_name,max_channels=max_channel)
                        #dream_flats.figure_img_flat_hist()

