import math
import numpy as np
from matplotlib import patches

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return n//a,a


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

def d2u(s): #remove dot and place underscores
    return np.asarray([ss.replace('.', '_') for ss in s])