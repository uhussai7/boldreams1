from utils import * #layers_by_model,layer_shapes,unique_2d_layer_shapes,keyword_layers,channels_to_use,channel_summer
from dataHandling import dataFetcher
import matplotlib.pyplot as plt


#get the the noise
base_path=get_base_path('local')
df=dataFetcher(base_path)
df.fetch(upto=1)
noise=df.dic['voxel_ncsnr']


#choose the roi
config_name='bb-alexnet_upto-16_bbtrain-False_max-channels_100.json'#sys.argv[1]
#ROI=sys.argv[2] #lets do all Rois
obj_name='roi_spec'#sys.argv[2]

#get config (this should come from sys.argv)
base='/home/uzair/nvme/'
config=load_config(base+'configs/' + config_name)
freesurfer_path = base + '/nsd/freesurfer/'


#mkdir to store dream outputs for later visualization
dream_path=base+'/dreams/'
if not os.path.exists(dream_path):
    os.makedirs(dream_path)

#load the flatmap stuff
ref_nii=nib.load(base+ '/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/Kastner2015.nii.gz')
i,j,k=df.dic['x'],df.dic['y'],df.dic['z']
sub='subj01'

#load the image
rois_dic=df.dic['roi_dic_combined']
fig,ax=plt.subplots(1)
for ROI in ['floc-places']:#list(rois_dic.keys()):
    dream_flat_name=config_name.split('.json')[0]+'_roi-'+ROI+'_obj-'+obj_name+'_flat.png'
    dream_img_name=config_name.split('.json')[0]+'_roi-'+ROI+'_obj-'+obj_name+'_img.pt'
    dream_fmri_name=config_name.split('.json')[0]+'_roi-'+ROI+'_obj-'+obj_name+'_fmri.pt'
    dream=torch.load(base+'/dreams/'+dream_img_name)
    ax.imshow(dream[0].moveaxis(0,-1))
    fmri_dream=torch.load(base+'/dreams/'+dream_fmri_name)

    #projection, hacky business
    faces_nii=array_to_func1pt8(i,j,k,fmri_dream[0],ref_nii)
    faces_surf=func1pt8_to_surfaces(faces_nii,base+'/nsd/',sub,method='linear')[0]
    flat_hemis=get_flats(freesurfer_path,sub)
    make_flat_map(faces_surf,flat_hemis,base+'/dream_figures/'+dream_flat_name)#,vmin=-3,vmax=3.6)

    #plot with ROI outline
    fig, ax = plt.subplots()
    acc=plt.imread(base+'/dream_figures/'+dream_flat_name)
    H,L=acc.shape[:2]
    x_off=350
    y_off=200
    early_outline=plt.imread(base+'/overlays/'+ '/V1V2V3.png')
    #plt.figure()
    aa=ax
    aa.set_axis_off()
    aa.imshow(acc[y_off:,int(L/2)-x_off:int(L/2)+x_off])
    aa.imshow(early_outline[y_off:,int(L/2)-x_off:int(L/2)+x_off])