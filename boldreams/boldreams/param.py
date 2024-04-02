from __future__ import absolute_import, division, print_function

#from lucent.optvis.param.color import to_valid_rgb
import torch
from lucent.optvis.param.color import _linear_decorrelate_color

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_valid_rgb(image_f, decorrelate=False):
    def inner():
        image = image_f()
        if decorrelate:
            image = _linear_decorrelate_color(image)
        #return image#/image.max()#torch.sigmoid(image)#torch.clamp(image,0,1)#torch.sigmoid(image)
        return torch.nn.functional.sigmoid(image)
    return inner

def ref_image(ref_img):
    param_f = ref_pixel_image
    params, image_f = param_f(ref_img)
    output = to_valid_rgb(image_f, decorrelate=False)
    return params, output

def ref_pixel_image(ref_img):
    tensor = ref_img.to(device).requires_grad_(True)
    return [tensor], lambda: tensor