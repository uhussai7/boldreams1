import sys

import matplotlib.pyplot as plt

sys.path.append('../../..')
from utils import *
from dataHandling import dataFetcher
from torchvision import io
import glob
#make a class for each model will make things a bit easier


#get the the noise
base_path=get_base_path('local')
df=dataFetcher(base_path)
df.fetch(upto=16)

#we are loading dreams of 5 models
dreams_path=base_path+'/dreams/'
overlay_path=base_path+'/overlays/'

models=['alexnet']#,'vgg11','RN50x4_clip_relu3_last']
bbtrains=['True']#,'False']
max_channels=['1']
objs=['roi_ref_img','roi_ref_img_roi_target']

overlay_file_names=get_overlay_file_names()

ROIs=['LO','PHC','IPS','MT','MST']

#get the image inds

for ROI in ROIs: #overlay_file_names.keys():#overlay_file_names.keys():
    img_ind = img_by_roi()[ROI]
    fmri_signal = df.dic['train_vox'][img_ind]
    flat_maps=DreamFlatmaps(base_path,df.dic)
    gt_fmri_file_name= 'ROI-%s_gt_fmri.png'%ROI #only roi is needed
    flat_maps.make_signal_flatmap(fmri_signal,get_overlay_file_names()[ROI],gt_fmri_file_name)
    for bbtrain in bbtrains:
        for model in models:
            for max_channel in max_channels:
                if not (model == 'RN50x4_clip_relu3_last' and bbtrain == 'True'):
                    model_fmri_signal=torch.load(dreams_path +
                    '/ref_img/bb-%s_upto-16_bbtrain-%s_max-channels_%s_roi-%s_fmri_model.pt'%(model,bbtrain,max_channel,
                                                                                              ROI))
                    model_fmri_file_name='bb-%s_upto-16_bbtrain-%s_max-channels_%s_roi-%s_fmri_model.png'%(model,
                                                                                                          bbtrain,
                                                                                                           max_channel,
                                                                                                           ROI)
                    flat_maps.make_signal_flatmap(model_fmri_signal,get_overlay_file_names()[ROI],model_fmri_file_name)
                    for obj in objs:
                        print('------------------------------------------------------------')
                        print('ROI:%s, bbtrain:%s, model:%s,obj:%s'%(ROI,bbtrain,model,obj))
                        print('------------------------------------------------------------')
                        if not (model == 'RN50x4_clip_relu3_last' and bbtrain == 'True'):
                            #finally make ref_img flat map
                            file_name=('dream_model-'+model+'_bbtrain-'+bbtrain+ '_ROI-' + ROI +'_max_channels-'+max_channel+
                                       '_obj-'+obj +'.png')
                            #print(file_name)
                            flat_maps.make_dream_ref_img_flatmap(model,bbtrain,ROI,obj,overlay_file_names[ROI],
                                                                 out_filename=file_name)


# models=['alexnet','vgg11','RN50x4_clip_relu3_last']
# bbtrains=['False','False']
# max_channels=['100']
# ROI='floc-places'
# objs=['roi','roi_spec']
# flat_maps.make_dream_ref_img_flatmap(models[0],bbtrains[0],ROI,objs[0],get_overlay_file_names()[ROI],'test.png')


