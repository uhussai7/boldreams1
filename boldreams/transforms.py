import torch
import torchvision.transforms as T
from torch.nn import Module

class dynamicTransform(Module):
    def __init__(self,r,t,s):
        super(dynamicTransform, self).__init__()
        self.num_calls=0
        self.r=r
        self.s=s
        self.t=t

    def forward(self, img):
        h,w=img.shape[-2],img.shape[-1]
        f=int(h*0.2)
        ht,wt=h//2,w//2
        mask=torch.zeros_like(img)
        mask[:,:,ht-f:ht+f,wt-f:wt+f]=1
        self.num_calls+=1
        i=self.num_calls
        rr,tt,ss=self.r[i],self.t[i],self.s[i]
        self.xfm=T.RandomAffine([rr,rr],translate=[tt,tt],scale=[ss,ss])
        if i<100:
            img=img*mask
        return self.xfm(img)

