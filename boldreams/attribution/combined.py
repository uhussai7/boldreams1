'''
Here we need something that gives a summary of all the different kinds of attribution:
    1) Pixel attribution
    2) Feature attribution
    3) ---
'''

from models.fmri import prep_encoder,channel_stacker
import torch
from utils import channel_summer,model_save_path
from scipy.stats import rankdata
import numpy as np
from tqdm import tqdm


class filter_attribution:
    def __init__(self,config,df):
        self.config=config
        self.df=df
        self.p_enc=prep_encoder(config,df)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc_terms = self.p_enc.get_encoder_terms(device=self.device)
        self.load_checkpoint()

    def load_checkpoint(self):
        train_path = model_save_path(self.config)
        print("load model with path: ", train_path)
        checkpoint = torch.load(train_path)
        if self.config['train_backbone'] == True:
            print('Loading full model')
            self.enc_terms.load_state_dict(checkpoint, strict=True)
        else:
            print('Loading partial model')
            self.enc_terms.load_state_dict(checkpoint, strict=False)



    def terms(self,imgs,batch_size=2):
        #simply return the terms
        Nv,Ni,Nc=self.enc_terms.Nv,len(imgs),channel_summer(self.p_enc.channel_basis_2d)
        fmri_terms=np.zeros([Ni,Nc,Nv])
        for i in tqdm(range(0, len(imgs), batch_size)):
            fmri_terms[i:i+batch_size]=self.enc_terms(imgs[i:i + batch_size].to(self.device)).detach().cpu()
        return fmri_terms

    def terms_roi_mean_std(self,roi,imgs,batch_size=2):
        #simply return the terms
        Nv,Ni,Nc=self.enc_terms.Nv,len(imgs),channel_summer(self.p_enc.channel_basis_2d)
        fmri_terms_mean,fmri_terms_std=np.zeros([Ni,Nc]),np.zeros([Ni,Nc])
        for i in tqdm(range(0, len(imgs), batch_size)):
            fmri_terms_std[i:i+batch_size], fmri_terms_mean[i:i+batch_size]=torch.std_mean(self.enc_terms(imgs[i:i +batch_size].to(
                self.device)).detach().cpu()[:,:,roi],dim=-1)
        return fmri_terms_mean,fmri_terms_std

    def terms_roi_abs_mean(self,roi,imgs,batch_size=2):
        #simply return the terms
        Nv,Ni,Nc=self.enc_terms.Nv,len(imgs),channel_summer(self.p_enc.channel_basis_2d)
        fmri_terms=np.zeros([Ni,Nc])
        for i in tqdm(range(0, len(imgs), batch_size)):
            fmri_terms[i:i+batch_size]=self.enc_terms(imgs[i:i + batch_size].to(self.device)).detach().cpu()[:,:,
                                       roi].abs().mean(-1)
        return fmri_terms

    def mean_rank_of_filters_over_rois(self,roi_dic,out_file,imgs_inds=None,batch_size=2, data_type='train'):
        #return the mean rank of filter over rois for imgs
        imgs=self.get_imgs(imgs_inds,data_type)
        Nv,Ni,Nc,Nroi=self.enc_terms.Nv,len(imgs),channel_summer(self.p_enc.channel_basis_2d),len(roi_dic.keys())
        #dimensions
        ranks=np.zeros([Ni,Nc,Nroi])
        for i in tqdm(range(0,len(imgs),batch_size)):
            fmri_roi=[]
            for roi in roi_dic.values():
                fmri_roi.append(self.enc_terms(imgs[i:i+batch_size].to(self.device))[:,:,roi].detach().cpu().mean(-1))
            fmri_roi=torch.stack(fmri_roi,dim=-1)
            ranks[i:i+batch_size]=(rankdata(fmri_roi,axis=1)/Nc)
        with open(out_file,'wb') as f: ## have to save as dictionary
            np.save(f,ranks)
        return 1





    # def rank_of_fenc_terms=self.p_enc.get_encoder_terms(device=self.device)ilter(self,img_inds,roi=None,data_type='train'):
    #     ranks=
    def get_imgs(self,img_inds=None,data_type='train'):
        #helper to get imgs
        if data_type=='train':
            imgs= self.df.training_data_loader(image_size=self.p_enc.input_size[-2:]).dataset.tensors[0]
        else:
            imgs= self.df.validation_data_loader(image_size=self.p_enc.input_size[-2:]).dataset.tensors[0]
        if img_inds is None:
            return imgs
        else:
            return imgs[img_inds]



