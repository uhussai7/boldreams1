from torchvision.models import alexnet,inception_v3,resnet18,resnet34,resnet50,resnet101,resnet152,vgg11,vgg13,\
    vgg16,vgg19
from torchvision.models import AlexNet_Weights, Inception_V3_Weights, ResNet18_Weights, ResNet34_Weights, \
    ResNet50_Weights,ResNet101_Weights,ResNet152_Weights, VGG11_Weights, VGG13_Weights,VGG16_Weights, VGG19_Weights
import clip
import os
import torch
def save_models(model_list=None,model_weights=None,save_path=None,system='local'):
    if model_list==None:
        model_list=[alexnet,
                    inception_v3,
                    resnet18,resnet34,resnet50,resnet101,resnet152,
                    vgg11,vgg13, vgg16,vgg19]
        model_weights=[AlexNet_Weights,
                       Inception_V3_Weights,
                       ResNet18_Weights, ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights,
                       VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights]
    if save_path==None:
        if system=='local':
            save_path='/home/uzair/nvme/torch_models/'
        elif system=='cluster':
            save_path='/cluster/projects/uludag/uzair/torch_models/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    for m in range(0,len(model_list)):
        model=model_list[m]
        weights=model_weights[m]
        this_model=model(weights=weights.IMAGENET1K_V1)
        print('model is %s'%model.__name__)
        print('saving as %s'%save_path + model.__name__ +'.pt' )
        print('\n')
        torch.save(this_model,save_path + model.__name__ +'.pt' )

    clip_models=clip.available_models()
    for m in range(0,len(clip_models)):
        model_name=clip_models[m]
        if len(model_name.split('/'))>1:
            model_name=str(model_name.split('/')[0]) + '_' + str(model_name.split('/')[1])
        model=clip.load(clip_models[m])[0]
        print('model is %s' % model_name)
        print('saving as %s' % save_path + model_name + '_clip.pt')
        print('\n')
        torch.save(model,save_path + model_name + '_clip.pt')



save_models()