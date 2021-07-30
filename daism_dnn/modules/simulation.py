#######################################################################################################
##  Simulation of artificial bulk RNA-seq datasets basd on data augmentation for training DAISM-XMBD  ##
#######################################################################################################

import sys
import time
import numpy as np
import pandas as pd
import random
import scanpy as sc
import anndata
import math
import tqdm

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
        
def preprocess_purified(purepath,platform,mode,testexp,caliexp=None,califra=None):
    """
    Preprocess purified samples
    :param purepath: 
    :param platform: "R",RNA,"S"single cell RNA
    :param mode: 
    :param caliexp:
    :param califra:
    :return:
    """
    raw_input = sc.read_h5ad(purepath)
    
    # get available cell types of purified
    cell_list_available = list(raw_input.obs['cell.type'].unique())

    if mode == "daism":
        # get calibration cell types
        cell_list_cali = califra.index
    if mode == "generic":
        cell_list_cali = cell_list_available

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
    if mode == "daism":
        commongenes = list(set(list(caliexp.index)).intersection(list(C_df[cell_list[0]].index)).intersection(list(testexp.index)))
        caliexp = caliexp.reindex(commongenes)
    if mode =="generic":
        commongenes = list(set(list(testexp.index)).intersection(list(C_df[cell_list[0]].index)))
    
    for cell in cell_list:
            C_all[cell] = C_df[cell].reindex(commongenes).T.values
    
    if platform == "R":
        counts_per_cell_after = np.median(np.vstack(tuple(C_all.values())).sum(axis=1))

        for cell in cell_list:
            C_all[cell] = sc_norm(C_all[cell],counts_per_cell_after=counts_per_cell_after)        

    return(commongenes,caliexp,C_all) 



def daism_simulation(trainexp, trainfra,C_all, random_seed, N,platform,min_f=0.01, max_f=0.99):
    print('DAISM mixtures simulation start!')
    
    gn = trainexp.shape[0] # get feature gene number
    cn = trainfra.shape[0] # get cell type number
    sn = trainfra.shape[1] # get sample number

    available_celltypes = list(C_all.keys())
    celltypes = list(trainfra.index)
    
    if N % sn == 0:
        n = int(N/sn) 
    else:
        n = int(N/sn)+1
        
    random.seed(random_seed)
    np.random.seed(random_seed)

    mixsam = np.zeros(shape=(N, gn))
    mixfra = np.zeros(shape=(N, cn))

    # create mixtures loop
    with tqdm(total=N) as pbar:
        for i,sampleid in enumerate(trainfra.columns):

            for j in range(n):
                if i*n+j < N:
                    random.seed(random_seed+i*n+j)
                    np.random.seed(random_seed+i*n+j)

                    # The fraction of calibration samples in the simulation data
                    mix_fraction = round(random.uniform(min_f,max_f),8)

                    # The fractions of purified samples in the simulation data
                    fraction = np.random.dirichlet([1]*len(available_celltypes),1)*(1-mix_fraction)
                    complete_fraction=[0]*cn
                    for k, act in enumerate(available_celltypes):
                        idx = celltypes.index(act)
                        complete_fraction[idx] = fraction[0,k]

                    mixfra[i*n+j] = trainfra.T.values[i]*mix_fraction

                    # RNA-seq TPM + RNA-seq TPM
                    if platform == "R":
                        
                        mixsam[i*n+j] = np.array(trainexp.T.values[i])*mix_fraction
                        
                        for k, cell in enumerate(trainfra[sampleid].T.index):
                            if cell in (list(C_all.keys())):
                                pure = C_all[cell][random.randint(0,C_all[cell].shape[0]-1)]

                                mixsam[i*n+j] += np.array(pure)*complete_fraction[celltypes.index(cell)]
                                mixfra[i*n+j][k] += complete_fraction[celltypes.index(cell)]
                        mixsam[i*n+j] = 1e+6*mixsam[i*n+j]/np.sum(mixsam[i*n+j])
                        
                    # RNA-seq TPM + scRNA-seq or Microarray + scRNA
                    elif platform == "S":
                        cell_sum = 500

                        for k, cell in enumerate(trainfra[sampleid].T.index):
                            if cell in (list(C_all.keys())):
                                pure = C_all[cell][np.random.choice(range(C_all[cell].shape[0]),math.ceil(complete_fraction[celltypes.index(cell)]*cell_sum))].sum(axis=0)
                                if k==0:
                                    pure_sum = pure
                                else:
                                    pure_sum += pure

                                mixfra[i*n+j][k] += complete_fraction[celltypes.index(cell)]
                            else:
                                continue
                    
                        # generate mixtures
                        mix = pure_sum + mix_fraction*np.array(trainexp.T.values[i])
                        mixsam[i*n+j] = 1e+6*mix/np.sum(mix)

                    pbar.update(1)
                    time.sleep(0.0001)
    
    print('DAISM mixtures simulation finish!')

    mixsam = pd.DataFrame(mixsam.T,index = trainexp.index)
    mixfra = pd.DataFrame(mixfra.T,index = trainfra.index)
    
    feature = list(mixsam.index)
    feature.sort()
    
    mixsam = mixsam.reindex(feature) 
    celltypes = list(mixfra.index)

    # add calibration samples to training datasets directly
    mixsam.columns = range(mixsam.shape[1])
    mixfra.columns = range(mixfra.shape[1])
    caliexp = trainexp.reindex(mixsam.index)
    califra = trainfra.reindex(mixfra.index)
    drop_columns = mixsam.iloc[:,list(np.random.choice(range(mixsam.shape[1]),caliexp.shape[1],replace=False))].columns
    mixsam = mixsam.drop(columns = drop_columns)
    mixfra = mixfra.drop(columns = drop_columns)
    mixsam = pd.concat([mixsam,caliexp],axis=1)
    mixfra = pd.concat([mixfra,califra],axis=1)

    return (mixsam, mixfra, celltypes, feature)

def generic_simulation(C_all, random_seed, N, platform,commongenes):

    print('Generic mixtures simulation start!')
        
    random.seed(random_seed)
    np.random.seed(random_seed)

    # get cell type list
    celltypes = list(C_all.keys())
    celltypes.sort()
    
    mixsam = np.zeros(shape=(N, len(commongenes)))
    mixfra = np.zeros(shape=(N, len(celltypes)))
    
    # create mixtures loop
    with tqdm(total=N) as pbar:
        for j in range(N):
            random.seed(random_seed+j)
            np.random.seed(random_seed+j)

            if j % 2 == 0:
                sparse = True
            else:
                sparse = False

            if sparse:
                no_keep = np.random.randint(1, len(celltypes))
                keep = np.random.choice(list(range(len(celltypes))), size=no_keep, replace=False)
                available_celltypes = [celltypes[i] for i in keep]                
            else:
                available_celltypes = celltypes

            fraction = np.random.dirichlet([1]*len(available_celltypes), 1)*1
            complete_fraction=[0]*len(celltypes)
            for k, act in enumerate(available_celltypes):
                idx = celltypes.index(act)
                complete_fraction[idx] = fraction[0,k]

            # RNA-seq TPM + RNA-seq TPM
            if platform == "R":                
                c_all = {}
                for cell in celltypes:
                    c_all[cell] = C_all[cell][random.randint(0,C_all[cell].shape[0]-1)]*complete_fraction[celltypes.index(cell)]

                mix = np.vstack(tuple(c_all.values())).sum(axis=0)
                mixsam[j] = 1e+6*mix/np.sum(mix)
                mixfra[j] = np.array(complete_fraction)

                pbar.update(1)
                time.sleep(0.0001)
                
            # RNA-seq TPM + scRNA-seq or Microarray + scRNA
            elif platform == "S":
                cell_sum = 500

                c_all = {}
                
                for cell in celltypes:
                    c_all[cell] = C_all[cell][np.random.choice(range(C_all[cell].shape[0]),math.ceil(complete_fraction[celltypes.index(cell)]*cell_sum))].sum(axis=0)

                mix = np.vstack(tuple(c_all.values())).sum(axis=0)            
                mixsam[j] = 1e+6*mix/np.sum(mix)
                mixfra[j] = np.array(complete_fraction)

                pbar.update(1)
                time.sleep(0.0001)

    print('Generic mixtures simulation finish!')
    
    mixsam = pd.DataFrame(mixsam.T,index = commongenes)
    mixfra = pd.DataFrame(mixfra.T,index = celltypes) 
    
    return (mixsam, mixfra, celltypes, commongenes)
  