import torch
from torch.nn import Module
from torch.optim import Adam

class integrated_gradient(Module):
    """
    A simple module to compute integrated gradients
    """
    def __init__(self,feature_ext):
        """
        Initializer
        :param model: The wrapped fmri encoding model
        """
        super(integrated_gradient, self).__init__()
        self.feature_ext=feature_ext

    def forward(self,img_in,roi,I_0=None,steps=10):

        img=torch.clone(img_in.detach())
        img.requires_grad=True
        optimizer=Adam(lr=1e-2,params=[img])

        if I_0 is None:
            I_0=torch.zeros_like(img)
            I_0.requires_grad=True

        ig = []
        alphas = torch.linspace(0, 1, steps)
        for alpha in alphas:
            imgp=I_0 + alpha*(img-I_0)
            features=self.feature_ext(imgp)
            features[:,roi].mean().backward()
            ig.append(img.grad)

        return torch.stack(ig).detach().cpu().abs().mean([0,1])

class integrated_gradient_terms(Module):
    """
    A simple module to compute integrated gradients
    """
    def __init__(self,feature_ext):
        """
        Initializer
        :param model: The wrapped fmri encoding model
        """
        super(integrated_gradient_terms, self).__init__()
        self.feature_ext=feature_ext

    def forward(self,img_in,term,roi,vox=None,I_0=None,steps=10):

        img=torch.clone(img_in.detach())
        img.requires_grad=True
        optimizer=Adam(lr=1e-2,params=[img])

        if I_0 is None:
            I_0=torch.zeros_like(img)
            I_0.requires_grad=True

        ig = []
        alphas = torch.linspace(0, 1, steps)
        for alpha in alphas:
            imgp=I_0 + alpha*(img-I_0)
            features=self.feature_ext(imgp)
            if vox:
                features[:,term,roi][:,vox].backward()
            else:
                features[:, term, roi].mean(-1).backward()
            ig.append(img.grad)

        return torch.stack(ig).detach().cpu().abs().mean([0,1])

class integrated_gradient_vox(Module):
    """
    A simple module to compute integrated gradients
    """
    def __init__(self,feature_ext):
        """
        Initializer
        :param model: The wrapped fmri encoding model
        """
        super(integrated_gradient_vox, self).__init__()
        self.feature_ext=feature_ext

    def forward(self,img_in,roi,vox,I_0=None,steps=10):

        img=torch.clone(img_in.detach())
        img.requires_grad=True
        optimizer=Adam(lr=1e-2,params=[img])

        if I_0 is None:
            I_0=torch.zeros_like(img)
            I_0.requires_grad=True

        ig = []
        alphas = torch.linspace(0, 1, steps)
        for alpha in alphas:
            imgp=I_0 + alpha*(img-I_0)
            features=self.feature_ext(imgp)
            features[:,roi][:,vox].backward()
            ig.append(img.grad)
        return torch.stack(ig).detach().cpu().abs().mean([0,1])


class integrated_gradient_channel(Module):
    """
    A simple module to compute integrated gradients
    """
    def __init__(self,feature_ext):
        """
        Initializer
        :param model: The wrapped fmri encoding model
        """
        super(integrated_gradient_channel, self).__init__()
        self.feature_ext=feature_ext

    def forward(self,img_in,roi,channel,I_0=None,steps=10):

        img=torch.clone(img_in.detach())
        img.requires_grad=True
        optimizer=Adam(lr=1e-2,params=[img])

        if I_0 is None:
            I_0=torch.zeros_like(img)
            I_0.requires_grad=True

        ig = []
        alphas = torch.linspace(0, 1, steps)
        for alpha in alphas:
            imgp=I_0 + alpha*(img-I_0)
            features=self.feature_ext(imgp)
            features[roi][:,channel].mean().backward()
            ig.append(img.grad)

        return torch.stack(ig).detach().cpu().abs().mean([0,1])