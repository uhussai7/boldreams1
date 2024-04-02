import math
from torch.nn import Module,init,ParameterList,Linear,ModuleDict, FractionalMaxPool2d
import torch
from torch.nn.parameter import Parameter
from utils import channel_summer,layers_by_model,layer_shapes,unique_2d_layer_shapes,channels_to_use
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn.functional import relu

class prep_encoder:
    def __init__(self,config,df,roi=None):
        self.config=config
        backbone_name = self.config['backbone_name']
        base_path = self.config['base_path']
        self.df=df
        if roi is None:
            self.Nv=self.df.dic['Nv']
        else:
            self.Nv=roi.sum()
        self.roi=roi

        # add models path
        torch_models_path = base_path + '/torch_models/'
        self.config['models_path'] = torch_models_path

        if 'clip' in backbone_name:
            backbone = torch.load(torch_models_path + '_'.join(backbone_name.split('_')[:2]) + '.pt').visual
        else:
            backbone = torch.load(torch_models_path + backbone_name + '.pt')
        self.layers_2d, self.layers_1d, self.input_size = layers_by_model(backbone_name)  # +'_relu3_last')
        # update layers 2d
        # layers_2d = keyword_layers(get_graph_node_names(backbone)[0],['relu3'])
        #self.layer_shps = layer_shapes(backbone, backbone.named_modules(), self.input_size)
        self.net_f=create_feature_extractor(backbone,return_nodes=self.layers_2d).to(config['device'])#+layers_1d)
        self.layer_shps = layer_shapes(self.net_f, self.input_size)
        self.res, self.res_id, self.channels_2d = unique_2d_layer_shapes(self.layers_2d, self.layer_shps)
        self.make_channel_bases(roi=self.roi)

    def make_channel_bases(self,roi=None):
        self.train_data=self.df.training_data_loader(roi=roi,image_size=self.input_size[-2:])
        if self.config['max_filters']==-1:
            self.channel_basis_2d = {}
            self.channel_basis_1d = {}
            for key in self.layers_2d:
                n_filters=self.layer_shps[key][1]
                self.channel_basis_2d[key]=torch.arange(0,n_filters)
            if self.layers_1d[0] is not None:
                for key in self.layers_1d:
                    n_filters=self.layer_shps[key][1]
                    self.channel_basis_1d[key]=torch.arange(0,n_filters)
        else:
            self.channel_basis_2d, self.channel_basis_1d = channels_to_use(self.layers_2d, self.net_f,
                                                                 self.train_data.dataset.tensors[0],
                                                                 device=self.config['device'],
                                                                 max_channels=self.config['max_filters'],
                                                                 max_percent=self.config['max_percent'])

    def get_encoder(self,device,multi_gpu=False):
        return encoder(self.net_f.float().to(device),self.Nv,self.channel_basis_2d,
        self.layers_2d,self.res,self.res_id,channels_1d=None,
        layers_1d=None,input_size=self.input_size).to(device)

    def get_encoder_multi_gpu(self,device1,device2):
        return encoder_multi_gpu(self.net_f.float(), self.Nv, self.channel_basis_2d,
                                 self.layers_2d, self.res, self.res_id, channels_1d=None,
                                 layers_1d=None, input_size=self.input_size,
                                 device1=device1,device2=device2)

    def get_encoder_terms(self,device):
        return encoder_terms(self.net_f.float().to(device), self.Nv, self.channel_basis_2d,
                       self.layers_2d, self.res, self.res_id, channels_1d=None,
                       layers_1d=None, input_size=self.input_size).to(device)

def channel_stacker(channels):
    layers_stack=[]
    channels_stack=[]
    for key in channels.keys():
        for channel in channels[key]:
            layers_stack.append(key)
            channels_stack.append(int(channel))
    return layers_stack,channels_stack

class encoder(Module):
    def __init__(self,model,Nv,channels_2d,layers_2d,res,res_id,
                 channels_1d=None,layers_1d=None,
                 input_size=[1,3,227,227]):
        super(encoder, self).__init__()
        self.Nv=Nv
        self.model=model #this should be the feature extractor
        self.layers_1d=layers_1d
        self.layers_2d=layers_2d
        self.channels_1d=channels_1d #these are a dictionary now, with inds of
        self.channels_2d=channels_2d #channels to take from each layer
        self.res=res
        self.res_id=res_id #this is dic of only 2d layers
        self.input_size=input_size
        self.device=list(model.parameters())[0].device


        #initialize rfs
        self.rfs=[]
        for res in self.res:
            self.rfs.append(Parameter(torch.empty(res+(self.Nv,),device=self.device)))
        self.rfs=ParameterList(self.rfs)

        #weights
        N_2d=channel_summer(channels_2d)
        self.w_2d=Parameter(torch.empty([N_2d,Nv],device=self.device))
        if layers_1d is not None:
            N_1d = channel_summer(channels_1d)
            self.w_1d=Parameter(torch.empty([N_1d,Nv],device=self.device))

        #bias
        self.b = Parameter(torch.empty(Nv))

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.b)
        if self.layers_1d is not None:
            init.kaiming_uniform_(self.w_1d,a=math.sqrt(5))
        init.kaiming_uniform_(self.w_2d,a=math.sqrt(5))
        for i in range(0,len(self.rfs)):
            init.kaiming_uniform_(self.rfs[i],a=math.sqrt(5))

    def forward(self,x):
        features=self.model(x)

        sigma_2d=torch.cat([torch.matmul(features[key][:,self.channels_2d[key]].flatten(-2,-1),
                                         self.rfs[self.res_id[key]].flatten(0,1)) for key in self.layers_2d],1)
        #sigma_2d = relu(sigma_2d)
        sigma_2d=(sigma_2d*self.w_2d[None,:,:]).sum(1)

        if self.layers_1d is not None:
            sigma_1d=(torch.cat([features[key][:,self.channels_1d[key]] for key in self.layers_1d],1)[:,:,
                      None]*self.w_1d[None,:,:]).sum(1)
            return sigma_2d + sigma_1d + self.b
        else:
            return sigma_2d + self.b

class encoder_multi_gpu(Module):
    def __init__(self,model,Nv,channels_2d,layers_2d,res,res_id,
                 channels_1d=None,layers_1d=None,
                 input_size=[1,3,227,227],device1='cuda:0',device2='cuda:1'):
        super(encoder_multi_gpu, self).__init__()
        self.Nv=Nv
        self.model=model #this should be the feature extractor
        self.layers_1d=layers_1d
        self.layers_2d=layers_2d
        self.channels_1d=channels_1d #these are a dictionary now, with inds of
        self.channels_2d=channels_2d #channels to take from each layer
        self.res=res
        self.res_id=res_id #this is dic of only 2d layers
        self.input_size=input_size
        self.device1=device1
        self.device2=device2

        self.model=self.model.to(self.device1)
        #initialize rfs
        self.rfs=[]
        for res in self.res:
            self.rfs.append(Parameter(torch.empty(res+(self.Nv,),device=self.device2)))
        self.rfs=ParameterList(self.rfs)

        #weights
        N_2d=channel_summer(channels_2d)
        self.w_2d=Parameter(torch.empty([N_2d,Nv],device=self.device2))
        if layers_1d is not None:
            N_1d = channel_summer(channels_1d)
            self.w_1d=Parameter(torch.empty([N_1d,Nv],device=self.device2))

        #bias
        self.b = Parameter(torch.empty(Nv,device=self.device2))

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.b)
        if self.layers_1d is not None:
            init.kaiming_uniform_(self.w_1d,a=math.sqrt(5))
        init.kaiming_uniform_(self.w_2d,a=math.sqrt(5))
        for i in range(0,len(self.rfs)):
            init.kaiming_uniform_(self.rfs[i],a=math.sqrt(5))

    def forward(self,x):
        features=self.model(x.to(self.device1))
        for key in features.keys():
            features[key] = features[key].to(self.device2)

        sigma_2d=torch.cat([torch.matmul(features[key][:,self.channels_2d[key]].flatten(-2,-1),
                                         self.rfs[self.res_id[key]].flatten(0,1)) for key in self.layers_2d],1)
        #sigma_2d = relu(sigma_2d)
        sigma_2d=(sigma_2d*self.w_2d[None,:,:]).sum(1)

        if self.layers_1d is not None:
            sigma_1d=(torch.cat([features[key][:,self.channels_1d[key]] for key in self.layers_1d],1)[:,:,
                      None]*self.w_1d[None,:,:]).sum(1)
            return (sigma_2d + sigma_1d + self.b).to(self.device1)
        else:
            return (sigma_2d + self.b).to(self.device1)

class encoder_terms(encoder):
    def __init__(self,model,Nv,channels_2d,layers_2d,res,res_id,
                 channels_1d=None,layers_1d=None,
                 input_size=[1,3,227,227]):
        super(encoder_terms,self).__init__(model,Nv,channels_2d,layers_2d,res,res_id,
                 channels_1d=channels_1d,layers_1d=layers_1d,
                 input_size=input_size)

    def forward(self,x):
        features = self.model(x)

        sigma_2d = torch.cat([torch.matmul(features[key][:, self.channels_2d[key]].flatten(-2, -1),
                                           self.rfs[self.res_id[key]].flatten(0, 1)) for key in self.layers_2d], 1)
        #sigma_2d = relu(sigma_2d)
        sigma_2d = (sigma_2d * self.w_2d[None, :, :])
        if self.layers_1d is not None:
            sigma_1d = (torch.cat([features[key][:, self.channels_1d[key]] for key in self.layers_1d], 1)[:, :,
                        None] * self.w_1d[None, :, :])
            return sigma_2d + sigma_1d
        else:
            return sigma_2d


