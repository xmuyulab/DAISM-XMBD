#######################################################################################################
##  Simulation of artificial bulk RNA-seq datasets basd on data augmentation for training DAISM-DNN  ##
#######################################################################################################

import sys
import time
import numpy as np
import pandas as pd
import random
from random import randint, sample
import scanpy as sc
import anndata
import math

def sc_norm(exp,counts_per_cell_after):
    """
    Normalize single cell RNA-seq data
    :param exp:
    :param counts_per_cell_after:
    :return:
    """
    adata = anndata.AnnData(exp)
    sc.pp.normalize_per_cell(adata,counts_per_cell_after=counts_per_cell_after)

    return np.array(adata.X)

def readh5ad(raw_input,celltype):
    """
    Load purified expression profiles of corresponding cell types pf calibration samples for data augmentation
    :param raw_input: purified samples (h5ad)
    :param celltype: cell types of calibration samples
    :return:
    """

    #select purified expression profiles of corresponding cell types pf calibration samples
    raw_input_fil = raw_input[raw_input.obs['cell.type'].isin(celltype)].copy()
            
    tr_x = raw_input_fil.X.astype(np.double).transpose()
    tr_x = pd.DataFrame(tr_x,index=raw_input_fil.var_names)
    
    return tr_x

class ShowProcess():

    i = 0 
    max_steps = 0 
    max_arrow = 50 
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) 
        num_line = self.max_arrow - num_arrow 
        percent = self.i * 100.0 / self.max_steps 
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' +'('+'%d' % self.i+'/'+'%d' %self.max_steps+')'+ '\r' 
        
        sys.stdout.write(process_bar) 
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0
        
def preprocess_purified(purepath,platform,samexp,samfra):
    """
    Preprocess purified samples
    :param purepath: 
    :param platform: 
    :param samexp:
    :param samfra:
    :return:
    """
    raw_input = sc.read_h5ad(purepath)
    
    # get available cell types of purified
    cell_list_available = list(raw_input.obs['cell.type'].unique())

    # get calibration cell types
    cell_list_cali = samfra.index

    # get overlap between calibration cell types and available cell types
    cell_list = list(set(cell_list_available).intersection(list(cell_list_cali)))
    
    if len(cell_list)==0:
        print('Error: No matched cell types in purified samples!')
        sys.exit(1)

    C_df = {} #pandas dict
    C_all = {} #numpy dict
    for cell in cell_list:
        C_df[cell] = readh5ad(raw_input,[cell])

    # get overlap between signature genes and purified samples genes
    commongenes = list(set(list(samexp.index)).intersection(list(C_df[cell_list[0]].index)))
    samexp = samexp.reindex(commongenes)
    
    for cell in cell_list:
            C_all[cell] = C_df[cell].reindex(commongenes).T.values
    
    if platform == "Rs" or platform == "Ms":
        counts_per_cell_after = np.median(np.vstack(tuple(C_all.values())).sum(axis=1))

        for cell in cell_list:
            C_all[cell] = sc_norm(C_all[cell],counts_per_cell_after=counts_per_cell_after)        
    
    return(commongenes,samexp,C_all) 


def daism_simulation(trainexp, trainfra,C_all, random_seed, N, outdir,platform, marker,min_f=0.01, max_f=0.99):
    print('Mixtures simulation start!')
    
    gn = trainexp.shape[0] # get feature gene number
    cn = trainfra.shape[0] # get cell type number
    sn = trainfra.shape[1] # get sample number

    if N % sn == 0:
        n = int(N/sn)
    else:
        n = int(N/sn)+1
        
    random.seed(random_seed)
    np.random.seed(random_seed)

    process_bar = ShowProcess(N, 'Mixtures simulation finish!')

    mixsam = np.zeros(shape=(N, gn))
    mixfra = np.zeros(shape=(N, cn))
    
    # create mixtures loop
    for i,sampleid in enumerate(trainfra.columns):

        for j in range(n):
            if i*n+j < N:
                random.seed(random_seed+i*n+j)
                np.random.seed(random_seed+i*n+j)

                # The fraction of calibration samples in the simulation data
                mix_fraction = round(random.uniform(min_f,max_f),8)

                # The fractions of purified samples in the simulation data
                fraction = np.random.dirichlet([1]*cn,1)*(1-mix_fraction)
                
                mixfra[i*n+j] = trainfra.T.values[i]*mix_fraction

                # RNA-seq TPM + RNA-seq TPM
                if platform == "Rt":
                    
                    mixsam[i*n+j] = np.array(trainexp.T.values[i])*mix_fraction
                    
                    for k, celltype in enumerate(trainfra[sampleid].T.index):

                        pure = C_all[celltype][random.randint(0,C_all[celltype].shape[0]-1)]

                        mixsam[i*n+j] += np.array(pure)*fraction[0,k]
                        mixfra[i*n+j][k] += fraction[0,k]
                    mixsam[i*n+j] = 1e+6*mixsam[i*n+j]/np.sum(mixsam[i*n+j])
                    
                # RNA-seq TPM + scRNA-seq or Microarray + scRNA
                elif platform == "Rs" or platform =='Ms':
                    cell_sum = 500

                    cell_pure = math.ceil(cell_sum*(1-mix_fraction))
                    mix_pure = [0]*gn
                    for k, celltype in enumerate(trainfra[sampleid].T.index):
                        pure = C_all[celltype][np.random.choice(range(C_all[celltype].shape[0]),math.ceil(fraction[0,k]*cell_sum))].sum(axis=0)
                        if k==0:
                            pure_sum = pure
                        else:
                            pure_sum += pure

                        mixfra[i*n+j][k] += fraction[0,k]
                 
                    # generate mixtures
                    mix = pure_sum + mix_fraction*np.array(trainexp.T.values[i])
                    mixsam[i*n+j] = 1e+6*mix/np.sum(mix)

                process_bar.show_process()
                time.sleep(0.0001)
    
    mixsam = pd.DataFrame(mixsam.T,index = trainexp.index)
    mixfra = pd.DataFrame(mixfra.T,index = trainfra.index)
    
    feature = list(set(list(marker)).intersection(list(mixsam.index)))
    feature.sort()
    
    mixsam = mixsam.reindex(feature) 
    celltypes = list(mixfra.index)
    
    return (mixsam, mixfra, celltypes, feature)
