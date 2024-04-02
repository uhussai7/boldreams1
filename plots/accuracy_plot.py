import numpy as np
import matplotlib.pyplot as plt
import torch
from dataHandling import dataFetcher
from scipy.stats import ttest_ind

base='/home/uzair/nvme/'
# df=dataFetcher(base)
# df.fetch(upto=1)

def get_corr(backbone_name,train_or_not,Nf):
    corr_name= '/trainedEncoders/corrs/backbone_name-%s_UPTO-16_epochs-10_train_backbone-%s_max_filters' \
            '-False_max_percent-%s_trained_corr.pt' %(backbone_name,train_or_not,Nf)
    return torch.load(base+corr_name).cpu().numpy()

def get_corr_mean_std_error(backbone_name,train_or_not,Nf,ROI=None):
    corr=get_corr(backbone_name,train_or_not,Nf)
    if ROI:
        return corr[ROI].mean(),corr[ROI].std()
    else:
        return corr.mean(), corr.std()/np.sqrt(len(corr))

def corr_diff_cut_off(backbone_name1,backbone_name2,train_or_not1,train_or_not2,Nf1,Nf2,cutoff=0.2):
    corr1=get_corr(backbone_name1,train_or_not1,Nf1)
    corr2=get_corr(backbone_name2,train_or_not2,Nf2)
    inds=[]
    for v in range(len(corr1)):
        if min(corr1[v],corr2[v])>=cutoff:
            inds.append(v)
    inds=np.asarray(inds)
    diff=corr1[inds]-corr2[inds]
    return diff.mean(),diff.std()

def xmas_plot(corr1,corr2,ax):
    y=np.asarray([corr1, corr2]).max(0)
    x=corr1-corr2
    ax.hexbin(x,y,cmap='inferno', bins=30,mincnt=1)
    #ax.set_xlim([-0.6,0.6])

#plt params
#okay make the plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 22,
})

#lets make the histograms
fig,ax=plt.subplots(1,2,figsize=(2*8,8))
plt.subplots_adjust(hspace=0.3,wspace=0.3)
ax[0].hist(get_corr('alexnet','False',100),70,histtype='step',label='AlexNet Type I')
ax[0].hist(get_corr('vgg11','False',100),70,histtype='step',label='Vgg Type I')
ax[0].hist(get_corr('RN50_clip_add_last','False',100),70,histtype='step',label='RN50 Type I')
ax[0].legend()
ax[0].set_title('Correlation of Type I')
ax[0].set_xlabel('Correlation')
ax[0].set_ylim([0,1300])
ax[1].hist(get_corr('alexnet','True',100),70,histtype='step',label='AlexNet Type II')
ax[1].hist(get_corr('vgg11','True',100),70,histtype='step',label='Vgg Type II')
#ax[0].hist(get_corr('RN50_clip_add_last','False',100),70,histtype='step',label='RN50 Type I')
ax[1].legend()
ax[1].set_title('Correlation of Type II')
ax[1].set_xlabel('Correlation')
ax[1].set_ylim([0,1300])


#lets make the histograms change feature size
fig,ax=plt.subplots(1,2,figsize=(2*8,8))
ax[0].hist(get_corr('alexnet','False',25),70,histtype='step',label='AlexNet-25\% Type I')
ax[0].hist(get_corr('vgg11','False',10),70,histtype='step',label='Vgg-10\% Type I')
ax[0].hist(get_corr('RN50_clip_add_last','False',1),70,histtype='step',label='RN50-1\% Type I')
ax[0].legend()
ax[0].set_title('Correlation of Type I')
ax[0].set_xlabel('Correlation')
ax[0].set_ylim([0,1050])

ax[1].hist(get_corr('alexnet','True',100),70,histtype='step',label='AlexNet-100\% Type II')
ax[1].hist(get_corr('vgg11','True',1),70,histtype='step',label='Vgg-1\% Type II')
#ax[0].hist(get_corr('RN50_clip_add_last','False',100),70,histtype='step',label='RN50 Type I')
ax[1].legend()
ax[1].set_title('Correlation of Type II')
ax[1].set_xlabel('Correlation')
ax[1].set_ylim([0,1050])


#effect of feature space size
Nfs=[1,5,10,15,20,25,50,75,100]

fig,ax=plt.subplots(3,2)
diff_mean_std = np.asarray([corr_diff_cut_off('alexnet','alexnet','False','False',Nf,100,cutoff=0.35) for Nf in Nfs])
#ax[0].plot(Nfs,diff_mean_std[:,0],'black')
ax[0,0].errorbar(Nfs,diff_mean_std[:,0],diff_mean_std[:,1],diff_mean_std[:,1],'black',marker='x',capsize=2,
                label='AlexNet Type I')
ax[0,0].set_title('AlexNet Type I')
diff_mean_std = np.asarray([corr_diff_cut_off('alexnet','alexnet','True','True',Nf,100,cutoff=0.35) for Nf in Nfs])
ax[0,1].errorbar(Nfs,diff_mean_std[:,0],diff_mean_std[:,1],diff_mean_std[:,1],'black',marker='x',capsize=2,
               label='AlexNet Type II')
ax[0,1].set_title('AlexNet Type II')

diff_mean_std = np.asarray([corr_diff_cut_off('vgg11','vgg11','False','False',Nf,100,cutoff=0.35) for Nf in Nfs])
#ax[0].plot(Nfs,diff_mean_std[:,0],'black')
ax[1,0].errorbar(Nfs,diff_mean_std[:,0],diff_mean_std[:,1],diff_mean_std[:,1],'black',marker='x',capsize=2,
                label='Vgg Type I')
ax[1,0].set_title('Vgg Type I')
diff_mean_std = np.asarray([corr_diff_cut_off('vgg11','vgg11','True','True',Nf,100,cutoff=0.35) for Nf in Nfs])
ax[1,1].errorbar(Nfs,diff_mean_std[:,0],diff_mean_std[:,1],diff_mean_std[:,1],'black',marker='x',capsize=2,
               label='Vgg Type II')
ax[1,1].set_title('Vgg Type II')

diff_mean_std = np.asarray([corr_diff_cut_off('RN50_clip_add_last','RN50_clip_add_last','False','False',Nf,100,cutoff=0.35) for Nf in Nfs])
#ax[0].plot(Nfs,diff_mean_std[:,0],'black')
ax[2,0].errorbar(Nfs,diff_mean_std[:,0],diff_mean_std[:,1],diff_mean_std[:,1],'black',marker='x',capsize=2,
                label='RN50 Type I')
ax[2,0].set_title('RN50 Type I')
ax[2,1].set_axis_off()
[axx.set_ylim([-0.16,0.16]) for axx in ax.flatten()]
[axx.set_ylabel(r'$\langle\rho_i - \rho_{100}\rangle$') for axx in ax.flatten()]
[axx.set_xlabel(r'Percentage of filters/layer') for axx in ax.flatten()]


#
fig,axs=plt.subplots(2,3,figsize=(26,17))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for Nf,ax in zip(Nfs[0:6],axs.flatten()):
    xmas_plot(get_corr('alexnet','False',Nf),
          get_corr('alexnet','False',100),ax)
    ax.set_title(r'AlexNet (Type I), $\%$ of features$=$'+str(Nf))
    ax.grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]
[axx.set_ylabel(r'max $\left(\rho_i,\rho_{100}\right)$') for axx in axs.flatten()]
[axx.set_xlabel(r'$\rho_i - \rho_{100}$') for axx in axs.flatten()]

fig,axs=plt.subplots(2,3,figsize=(26,17))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for Nf,ax in zip(Nfs[0:6],axs.flatten()):
    xmas_plot(get_corr('alexnet','True',Nf),
          get_corr('alexnet','True',100),ax)
    ax.set_title(r'AlexNet (Type II), $\%$ of features$=$'+str(Nf))
    ax.grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]
[axx.set_ylabel(r'max $\left(\rho_i,\rho_{100}\right)$') for axx in axs.flatten()]
[axx.set_xlabel(r'$\rho_i - \rho_{100}$') for axx in axs.flatten()]

fig,axs=plt.subplots(2,3,figsize=(26,17))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for Nf,ax in zip(Nfs[0:6],axs.flatten()):
    xmas_plot(get_corr('vgg11','False',Nf),
          get_corr('vgg11','False',100),ax)
    ax.set_title(r'Vgg (Type I), $\%$ of features$=$'+str(Nf))
    ax.grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]
[axx.set_ylabel(r'max $\left(\rho_i,\rho_{100}\right)$') for axx in axs.flatten()]
[axx.set_xlabel(r'$\rho_i - \rho_{100}$') for axx in axs.flatten()]

fig,axs=plt.subplots(2,3,figsize=(26,17))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for Nf,ax in zip(Nfs[0:6],axs.flatten()):
    xmas_plot(get_corr('vgg11','True',Nf),
          get_corr('vgg11','True',100),ax)
    ax.set_title(r'Vgg (Type II), $\%$ of features$=$'+str(Nf))
    ax.grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]
[axx.set_ylabel(r'max $\left(\rho_i,\rho_{100}\right)$') for axx in axs.flatten()]
[axx.set_xlabel(r'$\rho_i - \rho_{100}$') for axx in axs.flatten()]

fig,axs=plt.subplots(2,3,figsize=(26,17))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for Nf,ax in zip(Nfs[0:6],axs.flatten()):
    xmas_plot(get_corr('RN50_clip_add_last','False',Nf),
          get_corr('RN50_clip_add_last','False',100),ax)
    ax.set_title(r'RN50 (Type I), $\%$ of features$=$'+str(Nf))
    ax.grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]
[axx.set_ylabel(r'max $\left(\rho_i,\rho_{100}\right)$') for axx in axs.flatten()]
[axx.set_xlabel(r'$\rho_i - \rho_{100}$') for axx in axs.flatten()]

#plot some histograms for RN50
fig,axs=plt.subplots(3,3)
for Nf,ax in zip(Nfs,axs.flatten()):
    ax.hist(get_corr('RN50_clip_add_last','True',Nf),60)


#we need to make some xmas plots to compare the "best" models
class model_param:
    def __init__(self,backbone_name,type,Nf):
        self.backbone_name=backbone_name
        self.type=type
        self.Nf=Nf
        self.corr=get_corr(backbone_name,type,Nf)
        self.title_dic={'alexnet':r'AlexNet','vgg11':r'Vgg','RN50_clip_add_last':r'RN50',
                        'True':r'II','False':r'I'}

    def title(self):
        return r'\_'.join([self.title_dic[self.backbone_name],self.title_dic[self.type],'%d'%(self.Nf)])

#we want to compare 5 models the are split 3+2 into two types
type1_models={'1':model_param('alexnet','False',25),
              '2':model_param('vgg11','False',10),
              '3':model_param('RN50_clip_add_last','False',1)}
type2_models={'1':model_param('alexnet','True',100),
              '2':model_param('vgg11','True',1)}

#make xmas plots oof each type
from itertools import combinations,product

#type1 inter-comparisons
combos= list(combinations([1,2,3],2))
fig,axs=plt.subplots(1,len(combos),figsize=(len(combos)*9,9))
for k,(t1,t2) in enumerate(combos):
    print(t1,t2)
    xmas_plot(type1_models[str(t1)].corr,type1_models[str(t2)].corr,axs[k])
    title = r'$-$'.join([type1_models[str(t1)].title(), type1_models[str(t2)].title()])
    axs[k].set_title(title)
    axs[k].grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]

#type2 inter-comparisons
combos= list(combinations([1,2],2))
fig,axs=plt.subplots(1,len(combos),figsize=(9,9))
for k,(t1,t2) in enumerate(combos):
    print(t1,t2)
    xmas_plot(type2_models[str(t1)].corr,type2_models[str(t2)].corr,axs)
    title=r'$-$'.join([type2_models[str(t1)].title(),type2_models[str(t2)].title()])
    axs.set_title(title)
    axs.grid()
axs.set_ylim([-0.1,0.8])
axs.set_xlim([-0.7,0.7])

#type1-2 comaprisons
combos=list(product([1,2,3],[1,2]))
fig,axs=plt.subplots(2,3,figsize=(3*9,2*9))
for k,(t1,t2) in enumerate(combos):
    print(t1,t2)
    xmas_plot(type1_models[str(t1)].corr,type2_models[str(t2)].corr,axs.flatten()[k])
    title=r'$-$'.join([type1_models[str(t1)].title(),type2_models[str(t2)].title()])
    axs.flatten()[k].set_title(title)
    axs.flatten()[k].grid()
[axx.set_ylim([-0.1,0.8]) for axx in axs.flatten()]
[axx.set_xlim([-0.7,0.7]) for axx in axs.flatten()]
