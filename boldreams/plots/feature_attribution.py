import numpy as np
import matplotlib.pyplot as plt
import torch
from attribution import filter_attribution
from dataHandling import dataFetcher
from utils.io_utils import load_config,model_save_path
from models.fmri import channel_stacker,prep_encoder
from utils import closestDivisors,pass_images
from matplotlib.colors import Normalize
from matplotlib import patches,cm
from lucent.optvis import render
from torchvision.transforms import RandomAffine

#need a function to make rectangles
def rectangles(ax,TL,d,sep,Nw,Nh,values,colormap):
    for i in range(0,Nw):
        for j in range(0,Nh):
            tl_x=TL[0]+i*(d+sep)
            tl_y=TL[1]+j*(d+sep)
            rect=patches.Rectangle((tl_x,tl_y),d,d,facecolor=colormap.to_rgba(values[i,j]),edgecolor=(0.5,0.5,0.5,0.3))
            ax.add_patch(rect)

#def hot_cold_features(ax,features):

def canvas(H,W):
    fig,ax=plt.subplots(figsize=(W/2,H/2))
    ax.set_xlim(0,W)
    ax.set_ylim(0,H)
    #ax.axis('off')
    return fig,ax

def d2u(s): #remove dot and place underscores
    return np.asarray([ss.replace('.', '_') for ss in s])

def plot_imgs(imgs,ax):
    e=0.05
    delta=1/3
    #ax.set_xlim(0,1/3)
    ax.set_ylim(0,1/3)
    x=[0,delta,2*delta,3*delta]
    y=[0.0,0.0,0,0]
    for i in range(0,3):
        ax_=ax_off(ax.inset_axes([x[i]+e/2,y[i],delta-e/2,delta-e/2],transform=ax.transData))
        ax_.imshow(imgs[i])

def ax_off(ax):
    ax.axis('off')
    return ax

def hot_cold_features_per_layer(values,cnn,layer_names,layer_stack,channel_stack,top_n=4,thresholds=(256,)):
    #get the top_n coldest and hottest channels per layer and visulize there features
    layer_features=[]
    for l in range(0,len(layer_names)):
        layer_name = layer_names[l]
        layer_inds = np.asarray(layer_stack) == layer_name
        vals=values[layer_inds]
        layers=layer_stack[layer_inds]
        channels=channel_stack[layer_inds]
        hot=vals.argsort()[::-1][:top_n]
        cold=vals.argsort()[:top_n]
        hot_imgs=[]
        cold_imgs=[]
        for i in range(0,top_n):
            lt=str(layers[hot[i]])
            ct=str(channels[hot[i]])
            _=render.render_vis(cnn,lt + ':' + ct,show_image=False,thresholds=thresholds)
            hot_imgs.append(_[0][0])
            #lt = str(layers[cold[i]])
            #ct = str(channels[cold[i]])
            #_ = render.render_vis(cnn, lt + ':' + ct, show_image=False,thresholds=thresholds)
            #cold_imgs.append(_[0][0])
        layer_features.append([hot_imgs,cold_imgs])
    return layer_features

def top_features(values,cnn,layer_stack,channel_stack,top_n=4,thresholds=512):
    #top_inds=np.argsort(np.abs(values))[::-1]
    top_inds = np.argsort(values)[::-1]
    imgs=[]
    jitter_only = [RandomAffine(4, translate=[0.01, 0.01])]  # ,scale=[1,1], fill=0.0)]
    for i in range(0,top_n):
        lt=str(layer_stack[top_inds[i]])
        ct=str(channel_stack[top_inds[i]])
        _=render.render_vis(cnn,lt + ':' + ct,show_image=False,thresholds=(thresholds,),transforms=jitter_only)
        imgs.append(_[0][0])
    return imgs

def imshow_border(ax,img,border_radius=0.1,lw=2,axis='off'):
    border_radius = border_radius
    x, y = 0.0 + border_radius, 0.0 + border_radius  # Coordinates of the lower-left corner
    width, height = 1 - 2 * border_radius, 1 - 2 * border_radius  # Width and height of the rectangle
    rounded_rectangle = patches.FancyBboxPatch((x, y), width, height, boxstyle=f"round, pad={border_radius}",
                                               edgecolor='black', facecolor='none', lw=lw, transform=ax.transData,
                                               clip_on=False)
    ax.add_patch(rounded_rectangle)
    ax.imshow(img,
              extent=[0, 1, 0, 1],
              clip_path=rounded_rectangle, alpha=1)
    ax.axis(axis)

def load_checkpoint(config,enc):
    train_path = model_save_path(config)
    print("load model with path: ", train_path)
    checkpoint = torch.load(train_path)
    if config['train_backbone'] == True:
        print('Loading full model')
        enc.load_state_dict(checkpoint, strict=True)
    else:
        print('Loading partial model')
        enc.load_state_dict(checkpoint, strict=False)
    return enc


#paths
base='/home/uzair/nvme/'
config=load_config(base+'/configs/alexnet_medium_bbtrain-on_max-channels-100.json')
df=dataFetcher(config['base_path'])

#jsut get one session ofr images
UPTO_=16 #warning: changing upto for speed, sample should be large enough
df.fetch(upto=UPTO_)

#rois
rois=df.dic['roi_dic_combined']

#get the correct encoder to get the ordering
p_enc = prep_encoder(config, df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = load_checkpoint(config,p_enc.get_encoder(device).float().cuda())

#traning imgs
imgs=p_enc.train_data.dataset.tensors[0]

#get the response to find most active images for an roi
roi_img_inds={}
ROIs=['V1','V2','V3','floc-faces','floc-places']
ROIs_nice=['V1','V2','V3','Faces','Places']
fmri_test = pass_images(imgs, enc, enc.Nv,batch_size=8)
for ROI in ROIs:
    print(ROI)
    roi_img_inds[ROI]=fmri_test[:,rois[ROI]].mean(-1).argsort().flip(0)
enc=''

#plt some ffa iamges
fig,ax=plt.subplots(25,len(ROIs))
for j in range(0,len(ROIs)):
    ax_=ax[:,j]
    [ax_[i].imshow(imgs[roi_img_inds[ROIs[j]][25+i]].moveaxis(0,-1)) for i in range(0,25)]
nice_inds={'V1':2,'V2':2,'V3':0,'floc-faces':6,'floc-places':0}

#attribution class
att=filter_attribution(config,df)

#get the terms
terms_mean_roi,terms_std_roi={},{}
for ROI in ROIs:
    terms_mean_roi[ROI],terms_std_roi[ROI]=att.terms_roi_mean_std(rois[ROI],imgs[roi_img_inds[ROI][nice_inds[
        ROI]]].unsqueeze(0))

#layer related variables
layer_names=att.p_enc.layers_2d
N_layers=len(att.p_enc.layers_2d)
layer_shapes_1d=att.p_enc.channels_2d
layer_shapes_2d={key:closestDivisors(value) for key,value in layer_shapes_1d.items()}
layer_stack, channel_stack = channel_stacker(att.p_enc.channel_basis_2d)
layer_stack, channel_stack = np.asarray(layer_stack), np.asarray(channel_stack)


#image dimensions
W=64
H=32
w=0.5*W/N_layers #width for each layer

#rect dimensions
d=0.15
sep=0.06
pad=0.7/2
left_pad=5
top_pad=5
#largest neuron dims
neuron_shapes=np.asarray(list(layer_shapes_2d.values()))
n_H=neuron_shapes[:,1].max()
n_W=neuron_shapes[:,0].sum()
v_pad=0.28
#features dims
fdim=7.5

#color maps
m =cm.ScalarMappable(cmap=cm.bwr, norm=Normalize(vmin=-1, vmax=1))

fig,ax=canvas(H,W)

img_H=n_H*(sep+d)

for r,ROI in enumerate(ROIs):
    #normalize the attribution terms
    terms_mean=terms_mean_roi[ROI]
    values=terms_mean[0]#[roi_img_inds[ROI][0]]
    values=values/values.max()
    print(values.max())
    ##get the image_feautrues
    #features=hot_cold_features_per_layer(values,att.enc_terms.model,d2u(layer_names),d2u(layer_stack),channel_stack)
    features=top_features(values,att.enc_terms.model,d2u(layer_stack),channel_stack,thresholds=(1024))

    width=n_H*(d+sep)+pad + left_pad
    for l in range(0,N_layers): #start with each layer
        layer_name=layer_names[l]
        layer_inds=np.asarray(layer_stack)==layer_name
        vals=values[layer_inds].reshape(layer_shapes_2d[layer_name])
        width_layer=(d+sep)*vals.shape[0]+pad/2
        TL=[ width,H-(n_H*(sep+d)+pad+v_pad)*(r+1)-top_pad]
        print(TL)
        rectangles(ax,TL,d,sep,vals.shape[0],vals.shape[1],vals,m) #I think may just histograms is fine
        width=width+width_layer

    #neurons attribution heading
    neurons_text_x=(n_W*(sep+d))/2+n_H*(sep+d)+left_pad
    neurons_text_y=H-0.95*top_pad
    plt.text(neurons_text_x,neurons_text_y,'Channel attributions',
             horizontalalignment='center',verticalalignment='center',fontsize=18)

    #ROI heading
    plt.text(left_pad*0.73,H-(img_H+pad+v_pad)*(r+1/2)-top_pad,ROIs_nice[r],horizontalalignment='center',
             verticalalignment='center',
             fontsize=18)

    #inputs headding
    plt.text(left_pad+img_H/2,neurons_text_y,'Input',
             horizontalalignment='center',verticalalignment='center',fontsize=18)

    #place the images
    img_H=n_H*(d+sep)
    ax_ = ax.inset_axes([left_pad, H - (img_H + pad + v_pad) * (r + 1) - top_pad, img_H, img_H], transform=ax.transData)
    ax_.imshow(imgs[roi_img_inds[ROI][nice_inds[ROI]]].moveaxis(0,-1))
    ax_.axis('off')

    offset=width*0.035
    #decay heading (was hist before)
    plt.text(width +offset + img_H / 2, neurons_text_y, 'Decay',
             horizontalalignment='center', verticalalignment='center', fontsize=18)

    ax_hist=ax.inset_axes([width+offset,H-(img_H+pad+v_pad)*(r+1)-top_pad,img_H,img_H],transform=ax.transData)
    ax_hist.semilogy(np.sort(np.abs(values)/np.abs(values).max())[::-1][0:50])
    ax_hist.set_ylim(0.05,1.02)
    width = width+ offset + img_H + pad

    #feature visulization heading
    plt.text(width + img_H *3/2, neurons_text_y, 'Feature Visualizations',
             horizontalalignment='center', verticalalignment='center', fontsize=18)

    for i in range(0,3):
        ax_=ax.inset_axes([width,H-(img_H+pad +v_pad)*(r+1)-top_pad,img_H,img_H],transform=ax.transData)
        #ax_.imshow(features[i])
        ax_.axis('off')
        imshow_border(ax_,features[i],lw=3)
        width = width + img_H + pad

    # ax_=ax_off(ax.inset_axes([w*l + w/2-fdim/2,21,fdim,fdim/2],transform=ax.transData))
    # plot_imgs(features[l][0],ax_)

#okay so it seems we have a decent plot here, just need to figure out how to align stuff properly




# terms_avg_roi=[]
# for roi_name in df.dic['group_names'][:-1]:
#     #print(roi_name)
#     terms_avg_roi.append(att.terms_roi_mean(rois[roi_name],imgs[0:100]))
#
# layer_stack, channel_stack = channel_stacker(att.p_enc.channel_basis_2d)
# for r,terms in enumerate(terms_avg_roi):
#     plt.figure()
#     terms=(terms).mean(0)
#     #terms=terms/terms.max()
#     layer_vals=[]
#     for layer_name in att.p_enc.layers_2d:
#         layer_inds=np.asarray(layer_stack)==layer_name
#         layer_vals.append(terms[layer_inds].mean())
#     plt.plot(layer_vals)
#     plt.title(df.dic['group_names'][:-1][r])



#feature visualizations
#_=render.render_vis(att.enc_terms.model.eval().cuda(),'features_2:0',show_image='False')

# fig,ax=plt.subplots()
#
# #color scheme
# colors=plt.cm.viridis(np.linspace(0,1,len(att.p_enc.layers_2d)))
# layer_stack,channel_stack=channel_stacker(att.p_enc.channel_basis_2d)
# start=0
# for l,layer in enumerate(att.p_enc.layers_2d):
#     layer_inds=np.asarray(layer_stack)==layer
#     end=start+terms[0,layer_inds].shape[0]
#     print(start,end)
#     #ax.bar(np.arange(start,end), terms2[0][layer_inds][:,rois['V1']].mean(-1),color=colors[l])
#     ax.bar(np.arange(start, end), terms[:,layer_inds].mean(0), color=colors[l])
#     start=end


# #do the filters have some nice features that allow for nice visualization?
# from matplotlib import patches,cm
# from matplotlib.colors import Normalize
# layer=att.p_enc.layers_2d[0]
# layer_inds = np.asarray(layer_stack) == layer
# values=terms[:,layer_inds].mean(0).reshape(8,8)
# values=values/values.max()
# m =cm.ScalarMappable(cmap=cm.bwr, norm=Normalize(vmin=-1.5, vmax=1.5))
# # Map the scalar values to colors using the colormap and normalization
# fig,ax=plt.subplots()
# for i in range(0,8):
#     for j in range(0,8):
#         v=values[i,j]
#         ax.add_patch(patches.Rectangle((i,j),1,1,color=m.to_rgba(v)))
# ax.set_xlim(0,8)
# ax.set_ylim(0,8)

