from dataHandling import dataFetcher
import sys
import torch
from utils import *
from attribution import max_activation_stim_fmri,integrated_gradient
from models.fmri import prep_encoder
from boldreams import dream_wrapper
import matplotlib.pyplot as plt
import cortex
from cortex.freesurfer import get_surf
from mayavi import mlab

sys.argv=sysargs(sys.argv,'alexnet',1,10,-1,'False')
#configure
config={ 'SYSTEM':'local',
         'backbone_name':sys.argv[1],
         'UPTO':int(sys.argv[2]),
         'epochs':int(sys.argv[3]),
         'max_filters':int(sys.argv[4]),
         'train_backbone': sys.argv[5],
         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
         }
config['base_path']=get_base_path(config['SYSTEM'])
# df=dataFetcher(config['base_path'])
# df.fetch(upto=config['UPTO'])
# p_enc=prep_encoder(config,df)



#
freesurfer_path='/home/uzair/nvme/nsd/freesurfer/'
#ref_path='/home/uzair/nvme/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/meanbeta.nii.gz'
#ref_path='/home/uzair/nvme/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/prf-eccrois.nii.gz'
#ref_path='/home/uzair/nvme/nsd/nsddata_betas_freesurfer/ppdata/subj01/func1pt6mm/betas_fithrf_GLMdenoise_RR'
ref_path='/home/uzair/nvme/nsd/freesurfer/subj01/mri/T1.nii.gz'
change_freesurfer_subjects(freesurfer_path)
cortex.freesurfer.import_subj("subj01",freesurfer_subject_dir=freesurfer_path)
cortex.freesurfer.import_flat("subj01",'full',clean=False,auto_overwrite=True)
# cortex.align.automatic('subj01','full',ref_path)


subject='subj01'
xfm='full'
# #load the gii and change it
# from cortex.formats import read_gii,write_gii
# gii_path=cortex.database.default_filestore + '/'+subject +'/surfaces/'
# hemis=['lh','rh']
# for hemi in hemis:
#     pts,polys=read_gii(gii_path+'flat_'+hemi+'.gii')
#     #write_gii(gii_path+'flat_'+hemi+'_.gii', pts=pts, polys=polys)
#     flat=pts
#     #flat = pts[:, [1, 0, 2]]
#     #flat[:, 1] = flat[:, 1]
#     #write_gii(gii_path+'flat_'+hemi+'.gii', pts=flat, polys=polys)
#     if hemi == 'rh':
#         mlab.triangular_mesh(flat[:,0]-500,flat[:,1],flat[:,2],polys)
#     else:
#         mlab.triangular_mesh(flat[:,0],flat[:,1],flat[:,2],polys)

# This creates a Volume object for our test dataset for the given subject
# and transform
# test_data=nib.load('/home/uzair/nvme/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR'
#                    '/betas_session01.nii.gz').get_fdata()[:,:,:,0]
test_nii=nib.load(ref_path)
test_data=test_nii.get_fdata()
test_data=np.flip(test_data,1)
#vol_data = cortex.Volume(test_data, subject, xfm)
vol_data = cortex.Volume(np.moveaxis(np.moveaxis(test_data,0,-1),0,1), subject, test_nii.affine)
qf=cortex.quickflat.make_figure(vol_data)
cortex.webshow(vol_data)
plt.show()

# test_data=nib.load('/home/uzair/nvme/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/prf-eccrois.nii.gz').get_fdata()[:,:,:]
# test_data=np.flip(test_data,-1)
# test_data=np.moveaxis(np.moveaxis(test_data,0,-1),0,1)
# oi_data = cortex.Volume(test_data, subject, xfm)
# cortex.add_roi(roi_data,'ecc')
# # #example from gallery
# from mayavi import mlab
# volume = cortex.Volume.random(subject=subject, xfmname='full')
#
# # Plot a flatmap with the data projected onto the surface
# # By default the ROIs and their labels will be displayed
# _ = cortex.quickflat.make_figure(volume)
# plt.show()
#
# # Turn off the ROI labels
# _ = cortex.quickflat.make_figure(volume, with_labels=False)
# plt.show()
#
# # Turn off the ROIs
# _ = cortex.quickflat.make_figure(volume, with_rois=False)
# plt.show()
#
# #
# #
# # # #debgugging
# pts_lh,polys_lh,_=get_surf('S1','lh','patch','full'+'.flat',freesurfer_subject_dir='/home/uzair/miniconda3/envs/nsd/share/pycortex/db')
# pts_rh,polys_rh,_=get_surf('subj01','rh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path)
# mlab.triangular_mesh(pts_rh[:,0],pts_rh[:,1],pts_rh[:,2],polys_rh)
# mlab.triangular_mesh(pts_lh[:,0]+500,pts_lh[:,1],pts_lh[:,2],polys_lh)
# #
# import cortex
# import numpy as np
# import matplotlib.pyplot as plt
# import nibabel as nib
#

# # test_data[test_data==0]=np.nan
# subject='subj01'
# xfm='full'
# # This creates a Volume object for our test dataset for the given subject
# # and transform
# test_data=nib.load('/home/uzair/nvme/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR'
#                    '/betas_session01.nii.gz').get_fdata()[:,:,:,0]
# test_data=np.flip(test_data,-1)
# vol_data = cortex.Volume(np.moveaxis(np.moveaxis(test_data,0,-1),0,1), subject, xfm)
# cortex.quickflat.make_figure(vol_data,recache=1)
# cortex.webshow(vol_data)
# plt.show()
#
# # Can also alter the minimum and maximum values shown on the colorbar
# # vol_data_thresh = cortex.Volume(test_data, subject, xfm, vmin=-1, vmax=1)
# # cortex.quickshow(vol_data_thresh)
# # plt.show()
#
# # If you have NaN values, those voxels show up transparent on the brain
# # test_data[10:15, :, :] = np.nan
# # vol_data_nan = cortex.Volume(test_data, subject, xfm)
# # cortex.quickshow(vol_data_nan)
# # plt.show()
#
# # # Now you can do arithmetic with the Volume
# # vol_plus = vol_data + 1
# # cortex.quickshow(vol_plus)
# # plt.show()
# #
# # # You can also do multiplication
# # vol_mult = vol_data * 4
# # cortex.quickshow(vol_mult)
# # plt.show()
#


# import clip
# import torch
# import matplotlib.pyplot as plt
# from torchvision.datasets import CIFAR100
# import os
#
# def imshow_tensor(img):
#     plt.figure()
#     plt.imshow((img.moveaxis(0,-1).detach().cpu()))
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model, preprocess = clip.load('RN50x4', device='cuda')
# sz=288
# # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
# #
# # # Prepare the inputs
# # image, class_id = cifar100[3639]
# # image_input = preprocess(image).unsqueeze(0).to(device)
# # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
# #
# # # Calculate features
# # with torch.no_grad():
# #     image_features = model.encode_image(image_input)
# #     text_features = model.encode_text(text_inputs)
# #
# # # Pick the top 5 most similar labels for the image
# # image_features /= image_features.norm(dim=-1, keepdim=True)
# # text_features /= text_features.norm(dim=-1, keepdim=True)
# # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# # values, indices = similarity[0].topk(5)
# #
# # # Print the result
# # print("\nTop predictions:\n")
# # for value, index in zip(values, indices):
# #     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
#
# #can we try feature visualization
# from torch.nn import Module
#
# # class yet_another_wrapper(Module):
# #     def __init__(self,img_enc,txt_enc,text):
# #         super(yet_another_wrapper, self).__init__()
# #         self.img_enc=img_enc
# #         self.txt_enc=txt_enc
# #         self.txt_features=txt_enc(text).detach()
# #         self.txt_features=self.txt_features.norm(dim=-1,keepdim=True)
# #     def forward(self,img):
# #         self.img_features=self.img_enc(img)
# #         self.img_features=self.img_features.norm(dim=-1, keepdim=True)
# #         similarity = (self.img_features @ self.txt_features.T)/10000
# #         return similarity
# # class hmmm(Module):
# #     def __init__(self,yaw):
# #         super(hmmm, self).__init__()
# #         self.yaw=yaw
# #     def forward(self,img):
# #         out=self.yaw(img)
# #         #print(out)
# #         return out
#
# class wrapper(Module):
#     def __init__(self, model,text):
#         super(wrapper, self).__init__()
#         self.model=model
#         self.text=text
#     def forward(self,img):
#         return self.model(img,self.text)[0][0].unsqueeze(0)
#
#
# # wrap=yet_another_wrapper(model.encode_image,model.encode_text,clip.tokenize(['a photo of cat'],['a photo of '
# #                                                                                                 'dog'],
# #                                                                             ['a photo of man']).cuda())
# # ywrap=hmmm(wrap)
# wmodel=wrapper(model,clip.tokenize(['a busy street in new york city',
#                                     'a happy face of a woman with blonde hair',
#                                     'a woman at the beach']).cuda())
#
# from lucent.optvis import render, param, objectives
# from lucent.optvis.objectives import wrap_objective,handle_batch
# from lucent.modelzoo.util import get_model_layers
# from torch.optim import Adam,SGD
# from torchvision.transforms import GaussianBlur,Compose,RandomAffine
# import matplotlib.pyplot as plt
# from boldreams.transforms import dynamicTransform
# from boldreams.objectives import clip_img_features
# from lucent.modelzoo.util import get_model_layers
# from boldreams.objectives import clip_img_features
#
# b=1
# maxx=400
# interval=maxx
# scale_cut=300
# r=1*torch.ones(maxx+1)
# t=0.07*torch.ones(maxx+1)
# s=1.0*torch.ones(maxx+1)
# s[:scale_cut]=0.36
# t[scale_cut:]=0.2
# N=int((maxx)/interval)
# thresholds=torch.linspace(0,maxx,N+1).int()
# #optimizer=lambda params: Adam(params,lr=8.0e-2)
# optimizer=lambda params: SGD(params,lr=0.4)
# param_f = lambda: param.image(sz, fft=True, decorrelate=True,sd=0.04, batch=b)
# jitter_only = [RandomAffine(3,translate=[0.01,0.01],scale=(0.2,1.2),fill=0.0),GaussianBlur(3,sigma=(0.1,0.2))]
# #jitter_only = [dynamicTransform(r,t,s)]#[RandomAffine(13,translate=[0.13,0.13],scale=[0.24,0.9], fill=0.0)]
# #obj = clip_img_features() #- 1e-5* diversity("roi_ffa") #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([-2]).cuda())
#
# # _=render.render_vis(ywrap.cuda().eval(),'yaw:0',param_f=param_f,transforms=jitter_only,
# #                     optimizer=optimizer,fixed_image_size=224,thresholds=(3150,),show_image=False)
# @wrap_objective()
# def test_obj(layer,n_channel):
#     def inner(model):
#         return -model(layer)[0][:,n_channel].mean()
#     return inner
#
# layer='model_visual_layer4_5_relu3:2516'
# obj=objectives.neuron('model_visual_layer4_5_relu3',2516)
# _=render.render_vis(wmodel.eval(),test_obj('model',0),param_f=param_f,transforms=jitter_only,
#                     optimizer=optimizer,fixed_image_size=sz,thresholds=thresholds,show_image=False)
# [plt.subplots()[1].imshow(_[0][i]) for i in range(0,b)]
#
#
# # from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
# # import torchextractor as fx
# # names=fx.list_module_names(model)
# # loi=[]
# # for name in names:
# #     ns=name.split('.')
# #     if len(ns)==2 and ns[0]=='visual' and len(ns[1].split('layer'))==2:
# #         print(name)
# #         loi.append(name)
# # extractor=fx.Extractor(model,loi)
# # from torchvision.models import vit_b_16, ViT_B_16_Weights
# # from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
# #
# # vit=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
# # vit_nodes=get_graph_node_names(vit)
# # import torch
# # from torchvision.models import resnet50
# # from torch.optim import Adam
# # from torchvision.models.feature_extraction import get_graph_node_names
# # from torchvision.models.feature_extraction import create_feature_extractor
# # from torchvision.models.detection.mask_rcnn import MaskRCNN
# # from torchvision.models.detection.backbone_utils import LastLevelMaxPool
# # from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
# # from torchvision.models import alexnet,AlexNet_Weights,resnet18,\
# #     ResNet18_Weights,resnet101,ResNet101_Weights,vgg11,VGG11_Weights
# # from utils import alexnet_layers
# #
# # anet=alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
# # train_nodes,eval_nodes=get_graph_node_names(anet)
# # l_2d,l_1d = alexnet_layers()
# # test=create_feature_extractor(anet, return_nodes=l_2d+l_1d)
# # out=test(torch.rand(1,3,227,227))
#
#
# # from dataHandling import dataFetcher
# # from models import simple_model
# # import torch
# # from lucent.optvis import render, param, objectives
# # from lucent.optvis.objectives import wrap_objective,handle_batch
# # from lucent.modelzoo.util import get_model_layers
# # from torch.optim import Adam
# # from torchvision.transforms import GaussianBlur,Compose,RandomAffine
# # import matplotlib.pyplot as plt
# # from boldreams.objectives import roi
# # from utils.roi_utils import  make_roi_filter_from_groups,max_activation_ordering
# # from dataHandling import train_stimuli_from_dic
# # from boldreams.wrappers import model_wrapper
# # from attribution import integrated_gradient
# # from visualization.images import *
#
#
#
#
# # #load data
# # df=dataFetcher()
# # df.fetch()
# # Xtrain=train_stimuli_from_dic(df)
# #
# # #load model
# # EPOCH = 2#6
# # check_path= df.nsd_root + 'nets/' + 'simple_apply' +'/UPTO-%d/' %4 #where to store
# # model=simple_model(df.dic['fmaps'], df.dic['viscort_roi'].sum())
# # checkpoint = torch.load(check_path + 'model_epoch-%d'%EPOCH+'_ROI-999_fmap-max')
# # model.load_state_dict(checkpoint['model_state_dict'])
# #
# # #rois
# # roi_dic=make_roi_filter_from_groups(df.dic['group_names'],df.dic['group'],df.dic['voxel_roi'][df.dic['viscort_roi']])
# #
# # #face_roi
# # face_rois=df.specify_rois(df.manual_roi_folder,['floc-faces.nii.gz'])
# # face_rois={'ffa': face_rois[key]>0 for key in ['floc-faces']}
# #
# # #make wrapper
# # dreamer=model_wrapper(model.cuda(),face_rois).cuda()
# #
# # #max activation inds
# # ffa_inds=max_activation_ordering(Xtrain,dreamer,'roi_ffa')
# #
# # #integrated gradients
# # ig=integrated_gradient(dreamer)
# # s=10
# # fig,ax=plt.subplots(s,s)
# # for i in range(0,int(s*s)):
# #     img=Xtrain[ffa_inds[i]]
# #     ig_threshold(img,ig(img.cuda(),'roi_ffa'),ax.flatten()[i],gain=4.5)
# #
# # #wrap model and see if _modules works correctly
# # from boldreams.objectives import *
# # optimizer=lambda params: Adam(params,lr=4e-3)
# # param_f = lambda: param.image(227, fft=True, decorrelate=True,sd=0.01, batch=1)
# # jitter_only = [RandomAffine(3,translate=[0.01,0.01],fill=0.0)]
# # obj = roi("roi_ffa") - 1e-4* diversity("roi_ffa") #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([-2]).cuda())
# #
# # _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
# #                     optimizer=optimizer,fixed_image_size=227,thresholds=(500,),show_image=False)
# # [plt.subplots()[1].imshow(_[0][i]) for i in range(0,8)]
#
# # import torch
# # from models.simple import simple_model
# # from extractData import get_data
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from PIL import Image
# # from surface import meshes
# # from skimage.filters import gabor_kernel
# # from scipy.fft import fft, fftfreq
# #
# #
# # #gabor functions
# # def gabor_img(x0,y0,freq,sigma_x=None,sigma_y=None,theta=0,out_size=[227,227,3],rgb_weights=[1,1,1],phase=0):
# #     test=np.real(gabor_kernel(freq,sigma_x=sigma_x,sigma_y=sigma_y,theta=theta,offset=phase))
# #     mask=np.real(gabor_kernel(0,sigma_x=sigma_x,sigma_y=sigma_y,theta=theta,offset=phase))
# #     mask[mask<1e-5]=0
# #     test=mask*test
# #     test=np.pad(test, (40, 40), constant_values=0)
# #     test=test+abs(test.max())
# #     test=test/test.max()
# #     img2=np.moveaxis(np.asarray([a*test + (1-a)*test.mean()*np.ones_like(test) for a in rgb_weights]),0,-1)
# #     #img2=0.5*np.ones_like(img2)+img2
# #     img1=0.5*np.ones(out_size)
# #     return np.clip(1.0*embed(img1,img2,x0,y0),a_max=1,a_min=0)
# #
# # def embed(img1,img2,x0,y0):
# #     #make a blackground larger than both images,
# #     bg=0.5*np.ones([img1.shape[0]+img2.shape[0],img1.shape[1]+img2.shape[1],3])
# #     img1_mask=np.copy(bg)
# #     img2_mask = np.copy(bg)
# #     #get center vectors
# #     bg_m = np.asarray(bg.shape[0:2])/2
# #     img1_m=np.asarray(img1.shape[0:2])/2
# #     img2_m=np.asarray(img2.shape[0:2])/2
# #
# #     #place img1 in center of blackground
# #     c1 = (bg_m-img1_m).astype(int)
# #     c2 = (bg_m+img1_m).astype(int)
# #     img1_mask[c1[0]:c2[0],c1[1]:c2[1]]=img1
# #     #the center of img2 is place at x0,y0 of img1
# #     #what is x0,y0 in bg?
# #     b=np.asarray(c1)+np.asarray([x0,y0])
# #     d1 = (b - img2_m).astype(int)
# #     d2 = (b + img2_m).astype(int)
# #     img2_mask[d1[0]:d2[0],d1[1]:d2[1]]=img2
# #     bg=(img1_mask+img2_mask)/2
# #
# #     return bg[c1[0]:c2[0],c1[1]:c2[1]]
# #
# # #gabor class
# # class gabor_experiment:
# #     def __init__(self,model):
# #         self.model=model
# #
# #     def gabor_rotation_experiment(self,x0,y0,sigma=15,freq=0.01,phase=0,thetas=[0],image_size=[227,227,3],
# #                                   rgb_weights=[1,1,1],path='/home/uzair/Documents'):
# #     #we will cycle through signals freqs and then thetas
# #         out=[]
# #         grey=0.5*torch.ones([1,3,227,227])
# #         fmri_gabor = []
# #         imgs = []
# #         for i in range(0, len(thetas)):
# #             #print(i)
# #             g = torch.from_numpy(gabor_img(x0, y0, freq, sigma, sigma, thetas[i], rgb_weights=rgb_weights,
# #                                            phase=phase)).float()
# #             imgs.append(Image.fromarray((g.numpy() * 255).astype(np.uint8)))
# #             g = g.moveaxis(-1, 0).reshape(1, 3, 227, 227)
# #             fmri_gabor.append(model(g.cuda()).detach().cpu() - model(grey.cuda()).detach().cpu())
# #         imgss = (img for img in imgs)
# #         img = next(imgss)
# #         img.save(fp=path+'/rotation_x0-%d_y0-%d_sigma-%d_freq-%.2f_phase-%.2f.gif' %(int(x0),int(y0),int(sigma),freq,
# #                                                                                      phase) ,
# #                  format='GIF',
# #                  append_images=imgss,
# #                  save_all=True, duration=50, loop=0, quality='maximum')
# #         return torch.concat(fmri_gabor)
# #
# #     def gabor_translation_experiment(self,x0,y0,sigma=15,freq=0.01,phase=0,theta=0,image_size=[227,227,3],
# #                                   rgb_weights=[1,1,1],path='/home/uzair/Documents'):
# #     #we will cycle through signals freqs and then thetas
# #         out=[]
# #         grey=0.5*torch.ones([1,3,227,227])
# #         fmri_gabor = []
# #         imgs = []
# #         for x in x0:
# #             for y in y0:
# #                 g = torch.from_numpy(gabor_img(x, y, freq, sigma, sigma, theta, rgb_weights=rgb_weights,
# #                                                phase=phase)).float()
# #                 imgs.append(Image.fromarray((g.numpy() * 255).astype(np.uint8)))
# #                 g = g.moveaxis(-1, 0).reshape(1, 3, 227, 227)
# #                 fmri_gabor.append(model(g.cuda()).detach().cpu() - model(grey.cuda()).detach().cpu())
# #         imgss = (img for img in imgs)
# #         img = next(imgss)
# #         img.save(fp=path+'/translation_sigma-%d_freq-%.2f_phase-%.2f.gif' %(int(sigma),freq,
# #                                                                                      phase) ,
# #                  format='GIF',
# #                  append_images=imgss,
# #                  save_all=True, duration=50, loop=0, quality='maximum')
# #         return torch.concat(fmri_gabor)
# #
# #
# # def fft_gabor(fmri_gabor,thetas,N):
# #     fff = fftfreq(N, np.abs(thetas.max()-thetas.min())/N)
# #     fmri_fft=fft(fmri_gabor)
# #     return fmri_fft,fff
# #
# # def plot_fft(fig,ax,s,thetas,ft,ff,N,**kwargs):
# #     #plt.plot(ff[:N//2-1],ft[:N//2-1], color=color)
# #     #plt.plot(ff[N // 2-1:], ft[N // 2-1:],color=color)
# #     ax[0].plot(thetas/np.pi, s, **kwargs)
# #     ax[1].plot(ff*(np.pi), np.abs(ft), **kwargs)
# #     plt.show()
# #
# # def extract_roi(scalar,roi,group_list):
# #     this_filter=np.asarray([a in group_list for a in roi])==1
# #     return scalar[this_filter]
# #
# # def grey_img(shp=(1,3,227,227)):
# #     return 0.5*torch.ones(shp)
# #
# # def signal_in_roi(fmri,orig_roi,dic):
# #     groups=dic['group']
# #     fmri_out=[]
# #     for g in groups:
# #         this_filter=torch.asarray([a in g for a in orig_roi])
# #         fmri_out.append(fmri[this_filter])
# #     return fmri_out
# #
# # def barplot_roi(fmri_by_roi,dic,ax):
# #     bars=[]
# #     for f in fmri_by_roi:
# #         bars.append(f.abs().mean())
# #     ax.bar(np.asarray(dic['group_names'])[0:-1],np.asarray(bars)[0:-1])
# #
# # #loading the model
# # ROI = 999 #set to 999 to take all ROIs > 0
# # UPTO = 4#int(sys.argv[1]) #this is the number of sessions
# # EPOCH = 2#6
# # SYSTEM = 'local'
# #
# # #paths
# # base_paths = {'local':'/home/uzair/nvme/',
# #               'cluster':'/cluster/projects/uludag/uzair/'}
# # base = base_paths[SYSTEM]
# # dict_path = base + 'nsd/dictionary/'
# # dict_path_file = base + 'nsd/dictionary/dict_%d.pkl' % UPTO #this may not always be available
# # nsd_root = base + '/nsd/' #nsd root
# # check_path = nsd_root + 'nets/' + 'simple_apply' +'/UPTO-%d/' %UPTO #where to store
# #
# # #get the validation data
# # dic = get_data(nsd_root=nsd_root,UPTO=UPTO)
# # #check if ROI is all
# # if ROI==999: #this is for all ROIS>0
# #     orig_voxel_roi=np.copy(dic['voxel_roi'])
# #     dic['voxel_roi'][dic['voxel_roi']>=1]=999
# #     orig_voxel_roi=orig_voxel_roi[dic['voxel_roi']==ROI]
# #
# # #load the model
# # model = simple_model(dic['fmaps'],(dic['voxel_roi']==ROI).sum()).cuda()
# # checkpoint = torch.load(check_path + 'model_epoch-%d'%EPOCH+'_ROI-999_fmap-max')
# # model.load_state_dict(checkpoint['model_state_dict'])
# #
# # Xval=torch.from_numpy(dic['val_stim_single'])
# # Yval=torch.from_numpy(dic['val_vox_single'])
# # #Y_out=torch.zeros_like(Y[:,dic['voxel_roi']==ROI])
# #
# # X=torch.from_numpy(dic['train_stim'])
# # Y=torch.from_numpy(dic['train_vox'])[:,dic['voxel_roi']==ROI]
# # #Y_out=torch.zeros_like(Y[:,dic['voxel_roi']==ROI])
# #
# # x=dic['x'][dic['voxel_roi']==ROI]
# # y=dic['y'][dic['voxel_roi']==ROI]
# # z=dic['z'][dic['voxel_roi']==ROI]
# #
# # T1_file='/home/uzair/nvme/nsd/freesurfer/subj01/mri/T1.mgz'
# # surf_path='/home/uzair/nvme/nsd/freesurfer/subj01/surf/'
# # xfm_file='/home/uzair/nvme/nsd/nsddata/ppdata/subj01/transforms/anat0pt8-to-func1pt8.nii.gz'
# # mask_file='/home/uzair/nvme/nsd/nsddata/ppdata/subj01/anat/brainmask_0pt8.nii.gz'
# #
# # def surfaces():
# #     pial_l = meshes.Surface(T1_file,surf_path,'lh','equi0.5.pial')
# #     inf_l = meshes.Surface(T1_file,surf_path,'lh','inflated')
# #     pial_r = meshes.Surface(T1_file,surf_path,'rh','equi0.5.pial')
# #     inf_r= meshes.Surface(T1_file,surf_path,'rh','inflated')
# #     return pial_l,pial_r,inf_l,inf_r
# #
# # def signal_in_roi(fmri,orig_roi,dic):
# #     groups=dic['group']
# #     fmri_out=[]
# #     for g in groups:
# #         this_filter=torch.asarray([a in g for a in orig_roi])
# #         fmri_out.append(fmri[this_filter])
# #     return fmri_out
# #
# # def inertia_tensor(m,x,y,z):
# #     print(m.shape)
# #     I=np.zeros([3,3])
# #     I[0, 0]=(m*(y*y+z*z)).mean()
# #     I[1, 1]=(m*(x*x+z*z)).mean()
# #     I[2, 2]=(m*(x*x+y*y)).mean()
# #     I[0, 1]=-(m*x*y).mean()
# #     I[0, 2]=-(m*x*z).mean()
# #     I[1, 2]=-(m*y*z).mean()
# #     I[1, 0]=I[0, 1]
# #     I[2, 0]=I[0, 2]
# #     I[2, 1]=I[1, 2]
# #     return I
# #
# #
# # #lets begin by normalizing the data
# # Y_norm=(Y - Y.mean(0))/Y.std(0)
# #
# # #construct a basis for the rois
# # roi_basis=signal_in_roi(np.arange(Y_norm.shape[-1]),orig_voxel_roi,dic)
# # Y_norm_roi=[Y_norm[:,r] for r in roi_basis]
# # x_roi=[x[r] for r in roi_basis]
# # y_roi=[y[r] for r in roi_basis]
# # z_roi=[z[r] for r in roi_basis]
# #
# # I=[]
# # for roi in range(0,len(roi_basis)):
# #     print(roi)
# #     I_this_roi=[]
# #     for s in range(0, Y.shape[0]):
# #         I_this_roi.append(inertia_tensor(Y_norm_roi[roi][s,:],x_roi[roi],y_roi[roi],z_roi[roi]))
# #     I.append(I_this_roi)