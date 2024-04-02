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
config_names=[#'bb-alexnet_upto-16_bbtrain-False_max-channels_100.json',
              'bb-alexnet_upto-16_bbtrain-True_max-channels_1.json',
              # 'bb-vgg11_upto-16_bbtrain-False_max-channels_100.json',
              # 'bb-vgg11_upto-16_bbtrain-True_max-channels_100.json',
              # 'bb-RN50x4_clip_relu3_last_upto-16_bbtrain-False_max-channels_100.json'
              ]

#config_name = sys.argv[1] #"bb-alexnet_upto-16_bbtrain-False_max-channels_100.json"#sys.argv[1]
for config_name in config_names:

    # get config (this should come from sys.argv)
    base = '/home/uzair/nvme/'#"/home/u2hussai/projects_u/data/"#'/home/uzair/nvme/' #
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
    # roi_img_ind={'V1':4870,  'V2':9807,  'V3':7268, 'V3ab':3186,
    #          'hV4':6565, 'VO':2361,  'LO':2221, 'PHC':2765,
    #          'IPS':3485, 'MT':4442, 'MST':514,  'floc-bodies':961,
    #          'floc-places':7969,'floc-words':3409,'prf-eccrois':9909,
    #              'floc-faces':5382}
    roi_img_ind=img_by_roi()

    # set up the encoder
    p_enc = prep_encoder(config, df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = p_enc.get_encoder(device)
    enc.load_state_dict(torch.load(model_save_path(config)), strict=False)
    rois_dic = df.dic['roi_dic_combined']

    for ROI in list(rois_dic.keys()):
        print("working on ROI:", ROI)
        ref_img = p_enc.train_data.dataset.tensors[0][roi_img_ind[ROI]].unsqueeze(0)
        fmri=enc(ref_img.cuda()).detach().cpu()
        fmri_name = config_name.split('.json')[0] + '_roi-' + ROI + '_fmri_model.pt'
        torch.save(fmri,dream_path+fmri_name)

