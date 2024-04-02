#here we have scripts that make corr plots comparing two models

import matplotlib.pyplot as plt 

def two_corr(corr1,corr2,ax,**kwargs):
    #corr1,corr2 have to be 1d arrays of same size
    y=0.5*(corr1+corr2)
    x=corr2-corr1
    y_min=y.min()
    y_max=y.max()
    ax.hexbin(x,y,**kwargs)
    ax.plot([0,0],[y_min,y_max],'r')

