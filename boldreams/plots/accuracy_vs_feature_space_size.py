from dataHandling import dataFetcher
from utils.io_utils import get_base_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#get the the noise
base_path=get_base_path('local')
df=dataFetcher(base_path)
df.fetch(upto=16)
noise=df.dic['voxel_ncsnr']

corr={}
#get the corrs
ps=[1,5,10,15,20,25,50,75,100]
for p in ps:
    base_corr_name='backbone_name-alexnet_UPTO-16_epochs-10_train_backbone-False_max_filters-False_max_percent' \
                   '-%d_trained_corr.npy'%p
    corr[p]=np.load(base_path + '/trainedEncoders/' + base_corr_name)


#make a flat map of the accuracy

    
#make a summary plot, take voxels where one of the two corrs is greater than 0.2
comp=100
N_above={}
means=[]
std=[]
for p in ps[:-1]:
    inds=[]
    for v in range(len(corr[p])):
        if min(corr[p][v],corr[comp][v])>=0.2:
            inds.append(v)
    N_above[p]=np.asarray(inds)
    diff=corr[p][inds]-corr[comp][inds]
    means.append(diff.mean())
    std.append(diff.std())#/len(inds))

matplotlib.pyplot.rcParams.update({
    "text.usetex": True,
    "font.size":13,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica"
})
plots_path=base_path + '/plots/'
fig,ax=plt.subplots()
#ax.plot(ps[:-1],means,'x')
ax.errorbar(ps[:-1],means,std,marker='x',ecolor='grey',capsize=2,color='black')
ax.set_ylim([-0.160,0.05])
ax.set_xlabel('Percentage of filters per layer')
ax.set_ylabel(r"$\rho_x- \rho_{100}$")
plt.savefig(plots_path+ 'accuracy_vs_features_all.png',dpi=600,bbox_inches='tight',pad_inches=0.2)



fig,ax=plt.subplots()
hb=ax.hexbin(corr[100]-corr[1],0.5*(corr[100]+corr[1]),gridsize=40,mincnt=5,cmap='gray')
plt.colorbar(hb)
ax.set_xlabel(r'$\rho_{100}-\rho_{10}$')
ax.set_ylabel(r'$(\rho_{100}+\rho_{10})/2$')
ax.plot([0,0],[-0.08,0.8],color='r')
ax.set_ylim([-0.08,0.7])
plt.savefig(plots_path+ 'xmas_100_1.png',dpi=600,bbox_inches='tight',pad_inches=0.2)
