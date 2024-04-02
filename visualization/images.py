import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def ax_():
    """
    :return: Returns a quick axis object in new figure
    """
    return plt.subplots()[1]

def img_move_axis(img):
    if len(img.shape)==3:
        return img.moveaxis(0,-1).detach().cpu()
    if len(img.shape) == 4:
        return img.mean(0).moveaxis(0, -1).detach().cpu()


def ig_contour(img,ig,ax,kernel_size=11,sigma=5.1,levels=10,
               colors='w',linewidth=1.5):
    ig=ig/ig.max()
    ig=transforms.GaussianBlur(kernel_size,sigma)(ig.reshape((1,1,)+ig.shape)).mean([0,1])
    ax.imshow(img_move_axis(img))
    ax.contour(ig,levels=levels,colors=colors,linewidths=linewidth)

def ig_threshold(img,ig,ax,kernel_size=11,sigma=5.1,gain=1.1):
    ig=gain*ig/ig.max()
    ig=transforms.GaussianBlur(kernel_size,sigma)(ig.reshape((1,1,)+ig.shape)).mean([0,1])
    ax.imshow(img_move_axis(img)*ig[:,:,None])
