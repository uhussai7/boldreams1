import numpy as np
import torch
from PIL import Image, ImageDraw

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(np.deg2rad(phi))
    y = rho * np.sin(np.deg2rad(phi))
    return (x, y)

class retino:
    def __init__(self,img_size=[227,227]):
        self.img_size=img_size
        self.img=Image.new("RGB",img_size,'grey')
        self.draw=ImageDraw.Draw(self.img)
        self.xm=img_size[0]/2
        self.ym=img_size[1]/2

    def car2img(self,x,y):
        return x+self.xm,self.ym-y

    def checker_wedge(self,r1,r2,t1,t2,Nr,Nt):
        #Nr number of rings
        #Nt numver of wedges
        dr=(r2-r1)/(Nr)
        dt=(t2-t1)/(Nt)
        colors=[(255,255,255),(0,0,0)]
        color_id=0
        for r in range(0,Nr):
            color_id = (color_id + 1) % 2
            for s in range(0,Nt):
                self.wedge(r1+r*dr,
                           r1+(r+1)*dr,
                           t1+s*dt,
                           t1+(s+1)*dt,
                           color=colors[color_id],
                           outline=colors[color_id])
                color_id=(color_id+1)%2

    def wedge(self,r1,r2,t1,t2,color=(255,255,255),outline=(255,255,255),steps=20):
        xy = []
        #make outer arc
        dt=(t2-t1)/(steps-1)
        for i in range(0,steps):
            xt,yt=pol2cart(r2,t1+dt*i)
            xy.append(self.car2img(xt,yt))
        #line
        xt,yt=pol2cart(r1,t2)
        xy.append(self.car2img(xt,yt))
        #inner arc backward
        for i in range(0,steps):
            xt,yt=pol2cart(r1,t2-i*dt)
            xy.append(self.car2img(xt,yt))
        #line
        xt, yt = pol2cart(r2, t1)
        xy.append(self.car2img(xt, yt))
        #draw
        self.draw.polygon(xy,fill=color,outline=outline)

# s=retino()
# s.checker_wedge(110,150,0,360,3,36)
# s.img.show()