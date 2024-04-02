from collections import OrderedDict
import torch
from tqdm import tqdm
from utils.io_utils import model_save_path


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

def layers_by_model(model_name):
    if model_name == 'RN50x4_clip':
        layers_2d=['layer1','layer2','layer3','layer4']
        layers_1d=[None]
        return layers_2d, layers_1d, [1,3,288,288]
    if model_name == 'RN50x4_clip_relu3_last':
        layers_2d=['layer1.3.relu3','layer2.5.relu3','layer3.9.relu3','layer4.5.relu3']
        layers_1d=[None]
        return layers_2d, layers_1d, [1,3,288,288]
    if model_name == 'RN50x4_clip_add_last':
        layers_2d=['layer1.0.add', 'layer1.1.add', 'layer1.2.add', 'layer1.3.add', 'layer2.0.add', 'layer2.1.add',
                   'layer2.2.add', 'layer2.3.add', 'layer2.4.add', 'layer2.5.add', 'layer3.0.add', 'layer3.1.add',
                   'layer3.2.add', 'layer3.3.add', 'layer3.4.add', 'layer3.5.add', 'layer3.6.add', 'layer3.7.add',
                   'layer3.8.add', 'layer3.9.add', 'layer4.0.add', 'layer4.1.add', 'layer4.2.add', 'layer4.3.add',
                   'layer4.4.add', 'layer4.5.add']
        layers_1d = [None]
        return layers_2d, layers_1d, [1, 3, 288, 288]
    if model_name == 'RN50_clip_add_last':
        layers_2d=['layer1.0.add', 'layer1.1.add', 'layer1.2.add', 'layer2.0.add', 'layer2.1.add', 'layer2.2.add',
                   'layer2.3.add', 'layer3.0.add', 'layer3.1.add', 'layer3.2.add', 'layer3.3.add', 'layer3.4.add',
                   'layer3.5.add', 'layer4.0.add', 'layer4.1.add', 'layer4.2.add']
        layers_1d = [None]
        return layers_2d, layers_1d, [1, 3, 224, 224]
    if model_name == 'alexnet':
        layers_2d=['features.2','features.5','features.7','features.9','features.12']
        layers_1d=[None]
        return layers_2d, layers_1d, [1,3,256,256]
    if model_name == 'vgg11':
        layers_2d=['features.2','features.5','features.7','features.10','features.12',
                   'features.15','features.17','features.20']
        layers_1d=[None]
        return layers_2d, layers_1d, [1,3,224,224]

def keyword_layers(layer_names,keywords,exact=False):
    if not exact:
        return [name for name in layer_names for keyword in keywords if keyword in name]
    if exact:
        return [name for name in layer_names for keyword in keywords if keyword==name]

def unique_2d_layer_shapes(layer_names,layer_shapes):
    """
    This function extracts information about unique 2d resolutions
    :param layer_names: list of layer_names to extract
    :param layer_shapes: dictionary of layer shapes
    :return: list of unique resolutions and a dictionary with index of unique res of each layer
    """
    #get the unique 2d sizes
    unique_sizes=[]
    channels=OrderedDict()
    for key in layer_names:
        val_=layer_shapes[key]
        if val_.__len__()==4:
            val=val_[-2:]
            channels[key]=val_[1]
            if val not in unique_sizes:
                unique_sizes.append(tuple(val))
    unique_sizes=order_size_array(unique_sizes)
    #get the index dictionary
    inds_dic=OrderedDict()
    for key in layer_names:
        val_=layer_shapes[key]
        if val_.__len__()==4:
            val=tuple(val_[-2:])
            check=torch.asarray([val==a for a in unique_sizes])
            ind=torch.where(check==True)[0]
            inds_dic[key]=int(ind)
    return unique_sizes,inds_dic,channels

def order_size_array(sizes):
    sizes_sum=torch.asarray([torch.asarray(s).sum() for s in sizes])
    return [sizes[i] for i in sizes_sum.argsort(descending=True)]

def layer_shapes(net,input_size):
    shapes=OrderedDict()
    device=list(net.parameters())[0].device
    _=net(torch.rand(input_size,device=device))
    for key,value in _.items():
        shapes[key]=value.shape
    return shapes

def pass_images(imgs,model,Nv,batch_size=4): #will just pass some images through the model
    out=torch.zeros(len(imgs),Nv)
    print('Making predictions from stimuli')
    for i in tqdm(range(0,len(imgs),batch_size)):
        out[i:i+batch_size]=model(imgs[i:i+batch_size].cuda()).detach().cpu()
    return out

# def get_shape(name,shapes):
#     def hook(module,input,output):
#         shapes[name]=output.shape
#     return hook

# def layer_shapes(model,layers_to_extract,input_size=[1,3,227,227]):
#     shapes=OrderedDict()
#     device=list(model.parameters())[0].device
#     for name,module in layers_to_extract:
#         module.register_forward_hook(get_shape(name,shapes))
#     test_img=torch.rand(input_size,device=device)
#     _=model(test_img)
#     return shapes

class trainer:
    def __init__(self,model,train_loader,optim,loss,max_epochs=1,scheduler=None):
        self.model=model
        self.train_loader=train_loader
        self.optim=optim
        self.loss=loss
        self.max_epochs=max_epochs
        # self.device=model.device
        self.scheduler=scheduler

    def fit(self,interval=20):
        train_loss=[]
        for epoch in range(0,self.max_epochs):
            running_loss=0.0
            epoch_loss=0.0
            for i, data in enumerate(self.train_loader):
                inputs,labels=data
                self.optim.zero_grad()
                #outputs=self.model(inputs.to(self.device))
                #loss=self.loss(outputs,labels.to(self.device))
                outputs=self.model(inputs.cuda())
                loss=self.loss(outputs,labels.cuda())
                #print(loss.device)
                loss.backward()
                self.optim.step()
                running_loss+=loss
                if i % interval == interval -1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / interval:.3f}')
                    running_loss = 0.0
            self.scheduler.step()
        return self.model

# class ModuleHook: #adapted from lucent
#     """
#     A class to register forward hooks and save output of modules as features
#     """
#     def __init__(self, module):
#         """
#         Initializer
#         :param module: model to register hooks too
#         """
#         self.hook = module.register_forward_hook(self.hook_fn)
#         self.module = None
#         self.features = None
#
#     def hook_fn(self, module, input, output):
#         self.module = module
#         self.features = output
#
#     def close(self):
#         self.hook.remove()
#
# def hook_model(model,layers_to_hook=None): #adapted from lucent
#     """
#     Attaches hooks to all modules in the model
#     :param model: the model to attach hooks to
#     :return: hook function, hook names
#     """
#     features = OrderedDict()
#
#     def check_layer(layer_name,layers_to_hook):
#         #print(layer_name)
#         if layers_to_hook is None:
#             return True
#         elif layer_name in layers_to_hook:
#             #print('layer chosen', layer_name)
#             return True
#         else:
#             return False
#
#     def hook_layers(net,layers_to_hook=layers_to_hook, prefix=[]):
#         """
#         We go through all the modules recursively register hooks on these modules through ModuleHook class
#         :param net:
#         :param prefix:
#         :return:
#         """
#         if hasattr(net, "_modules"):
#             for name, layer in net._modules.items():
#                 if layer is None:
#                     continue
#                 if check_layer("_".join(prefix + [name]),layers_to_hook):
#                     features["_".join(prefix + [name])] = ModuleHook(layer)
#                 hook_layers(layer, prefix=prefix + [name])
#
#     hook_layers(model,layers_to_hook=layers_to_hook)
#
#     def hook(layer):
#         """
#         Returns the feature map of layer
#         :param layer: string for layer
#         :return: feature map
#         """
#         assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
#         out = features[layer].features
#         assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
#         return out
#
#     return hook
#
# def module_iter(layers,net,prefix=''):
#     if hasattr(net,'_modules'):
#         for name,layer in net._modules.items():
#             #print(prefix+name)
#             layers[prefix+'_'+name]=layer
#             module_iter(layers,layer,name)
#     return layers

def channel_activations(layers_2d,feature_extractor,imgs,layers_1d=[],batchsize=8,device='cpu'):
    N=imgs.shape[0]
    features=feature_extractor(imgs[:batchsize].to(device))
    outs_2d=dic_mean_2d(features,layers_2d)
    outs_1d={}
    if len(layers_1d)>0:
        for key in layers_1d:
            outs_1d[key]=features[key]
    imgs=imgs[batchsize:]
    print('Making predictions from stimuli')
    for i in tqdm(range(batchsize,N,batchsize)):
        features=feature_extractor(imgs[i:i+batchsize].to(device))
        for layer in layers_2d:
            outs_2d[layer]=torch.cat([outs_2d[layer],features[layer].mean([-2,-1]).detach().cpu()])
        for layer in layers_1d:
            outs_1d[layer]=torch.cat([outs_1d[layer],features[layer].detach().cpu()])
    return outs_2d,outs_1d

def channels_to_use(layers_2d,feature_extractor,imgs,layers_1d=[],batchsize=8,device='cpu',max_channels=512,
                    max_percent=None,filter_ranks=None):
    activations_2d,activations_1d=channel_activations(layers_2d,feature_extractor,imgs,layers_1d,batchsize,device)
    sd_2d=sorter(activations_2d,max_channels,max_percent)
    sd_1d=sorter(activations_1d,max_channels,max_percent)
    return sd_2d,sd_1d

def sorter(activations,max_channels,max_percent):
    sd = {}
    if max_percent is None:
        if max_channels == -1:
            for key in activations.keys():
                sd[key] = activations[key].std(0).sort(descending=True)[1]
        else:
            for key in activations.keys():
                aa = activations[key].std(0).sort(descending=True)[1]
                sd[key] = aa[:max_channels]
    else:
        for key in activations.keys():
            aa=activations[key].std(0).sort(descending=True)[1]
            max_channels=int(len(aa)*max_percent/100)
            sd[key] = aa[:max_channels]
    return sd

def channel_summer(sd):
    N=0
    for key in sd.keys():
        N+=len(sd[key])
    return N

def dic_mean_2d(dic,layers):
    for key in layers:
        dic[key]=dic[key].mean([-2,-1]).detach().cpu()
    return dic


#from lucent library
def get_model_layers(model, getLayerRepr=False):
    """
    If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
    If it's False, just return a list of layer names
    """
    layers = OrderedDict() if getLayerRepr else []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr:
                    layers["_".join(prefix+[name])] = layer.__repr__()
                else:
                    layers.append("_".join(prefix + [name]))
                get_layers(layer, prefix=prefix+[name])
    get_layers(model)
    return layers

def alexnet_layers():
    features=['features.2','features.5','features.7','features.9','features.12']
    classifiers=['classifier.3','classifier.5','classifier.6']
    return features, classifiers

def central_voxels(roi_dic,rfs0,split_factor=3):
    #roi_dic, rf0
    #get the shapes
    H=rfs0.shape[0]
    Nv=rfs0.shape[-1]

    #normalize the rfs
    rfs0=rfs0.view(-1,Nv).abs()
    rfs0=rfs0/rfs0.max()
    rfs0[rfs0<0.5]=0

    #create the central window
    central_window=torch.zeros(H,H)
    mid=int(H/2)
    delta=int(H/split_factor/2)
    central_window[mid-delta:mid+delta,mid-delta:mid+delta]=1
    vox_inds={}
    for roi_name,roi_filter in roi_dic.items():
        this_rf=rfs0[:,roi_filter]
        inds=torch.matmul(central_window.flatten(),this_rf).argsort(descending=True)
        vox_inds[roi_name]=inds

    return vox_inds

def img_by_roi():
    img_dic={'V1':4870,'V2':9807,'V3':7268,
             'V3ab':3186,'hV4':6565,
             'VO':2361,'LO':2221,
             'MST':514, 'MT':4442,
             'PHC':2046, 'IPS':3485,
             'floc-bodies':961,'floc-words':3409,'floc-places':7969,'floc-faces':5382,
             'prf-eccrois':9909}
    return img_dic

