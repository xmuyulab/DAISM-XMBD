#!/usr/bin/env python
import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
import random
from random import randint, sample
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from math import sqrt
from sklearn.metrics import mean_squared_error
import anndata
import scanpy.api as sc
import math
import simulation
import training
import prediction
from pandas.core.index import RangeIndex
import pandas.util.testing as tm
#--------------------------------------        
#--------------------------------------        

# main()

parser = argparse.ArgumentParser(description='DAISM-DNN deconvolution.')
subparsers = parser.add_subparsers(dest='subcommand', help='Select one of the following sub-commands')

parser_a = subparsers.add_parser('DAISM-DNN', help='DAISM-DNN')
#parser_a.add_argument("-cell", type=str, help="The mode of cell types, [C]: Coarse, [F]: Fine", default='C')
parser_a.add_argument("-platform", type=str, help="The platform of data, [Rs]: RNA-seq tpm + scRNA, [Rt]: RNA-seq tpm + tpm, [Ms]: Microarray + scRNA", default="Rs")
parser_a.add_argument("-caliExp", type=str, help="The calibration sample expression file", default="../data/testdata/example_refexp.txt")
parser_a.add_argument("-caliFra", type=str, help="The calibration sample ground truth file", default="../data/testdata/example_reffra.txt")
parser_a.add_argument("-pureExp", type=str, help="The purified expression", default="")
parser_a.add_argument("-simNum", type=int, help="The number of simulation sample ", default=16000)
parser_a.add_argument("-inputExp", type=str, help="The test sample expression file", default="../data/testdata/example_tarexp.txt")
parser_a.add_argument("-outputDir", type=str, help="The directory of output result file", default="../output/")

parser_b = subparsers.add_parser('simulation', help='simulation')
parser_b.add_argument("-platform", type=str, help="The platform of data, [Rs]: RNA-seq tpm + scRNA, [Rt]: RNA-seq tpm + tpm, [Ms]: Microarray + scRNA", default="Rs")
parser_b.add_argument("-caliExp", type=str, help="The calibration sample expression file", default="../data/testdata/example_refexp.txt")
parser_b.add_argument("-caliFra", type=str, help="The calibration sample ground truth file", default="../data/testdata/example_reffra.txt")
parser_b.add_argument("-pureExp", type=str, help="The purified expression", default="")
parser_b.add_argument("-simNum", type=int, help="The number of simulation sample ", default=16000)
parser_b.add_argument("-outputDir", type=str, help="The directory of output result file", default="../output/")



parser_c = subparsers.add_parser('training', help='training')
parser_c.add_argument("-trainExp", type=str, help="The simulated sample expression file", default="../data/testdata/example_refexp.txt")
parser_c.add_argument("-trainFra", type=str, help="The simulated sample ground truth file", default="../data/testdata/example_reffra.txt")
parser_c.add_argument("-outputDir", type=str, help="The directory of output result file", default="../output/")

parser_d = subparsers.add_parser('prediction', help='prediction')
parser_d.add_argument("-inputExp", type=str, help="The test sample expression file", default="../data/testdata/example_tarexp.txt")
parser_d.add_argument("-model", type=str, help="The deep-learing model file trained by DAISM-DNN", default="../output/dnn_daism_model.pkl")
parser_d.add_argument("-cellType", type=str, help="Model celltypes", default="../output/dnn_daism_celltypes.txt")
parser_d.add_argument("-feature", type=str, help="Model celltypes", default="../output/dnn_daism_celltypes.txt")
parser_d.add_argument("-outputDir", type=str, help="The directory of output result file", default="../output/")



def main():
    # parse some argument lists
    inputArgs = parser.parse_args()
    random_seed = 777
    if (inputArgs.subcommand=='DAISM-DNN'):
        min_f = 0.01
        max_f = 0.99
        lr = 1e-4
        batchsize = 32
        num_epoches = 500
        ncuda = 0
        caliexp = pd.read_csv(inputArgs.caliExp, sep="\t", index_col=0)
        califra = pd.read_csv(inputArgs.caliFra, sep="\t", index_col=0)
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.pureExp,inputArgs.platform,caliexp)
        mixsam, mixfra, celltypes, feature = simulation.daism_simulation(caliexp,califra,C_all,random_seed,inputArgs.simNum,inputArgs.outputDir,inputArgs.platform,commongenes,min_f,max_f)
        pd.DataFrame(feature).to_csv(inputArgs.outputDir+'/DAISM-DNN_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outputDir+'/DAISM-DNN_celltypes.txt',sep='\t')
        mixsam.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixfra.txt',sep='\t')
        
        model = training.dnn_training(mixsam,mixfra,random_seed,inputArgs.outputDir,num_epoches,lr,batchsize,ncuda)
        pd.DataFrame(list(mixfra.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_celltypes.txt',sep='\t')
        pd.DataFrame(list(mixsam.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_feature.txt',sep='\t')
        
        test_sample = pd.read_csv(inputArgs.inputExp, sep="\t", index_col=0)
        result = prediction.dnn_prediction(model, test_sample, list(mixfra.index), list(mixsam.index),inputArgs.outputDir,ncuda)
        result.to_csv(inputArgs.outputDir+'/DAISM-DNN_result.txt',sep='\t')
        
    if (inputArgs.subcommand=='simulation'):
        min_f = 0.01
        max_f = 0.99
        
        caliexp = pd.read_csv(inputArgs.caliExp, sep="\t", index_col=0)
        califra = pd.read_csv(inputArgs.caliFra, sep="\t", index_col=0)
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.pureExp,inputArgs.platform,caliexp)
       
        mixsam, mixfra, celltypes, feature = simulation.daism_simulation(caliexp,califra,C_all,random_seed,inputArgs.simNum,inputArgs.outputDir,inputArgs.platform,commongenes,min_f,max_f)
        pd.DataFrame(feature).to_csv(inputArgs.outputDir+'/DAISM-DNN_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outputDir+'/DAISM-DNN_celltypes.txt',sep='\t')
        mixsam.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixfra.txt',sep='\t')
    
    if (inputArgs.subcommand=='training'):
        lr = 1e-4
        batchsize = 32
        num_epoches = 500
        ncuda = 0
        mixsam = pd.read_csv(inputArgs.trainExp, sep="\t", index_col=0)
        mixfra = pd.read_csv(inputArgs.trainFra, sep="\t", index_col=0)
        model = training.dnn_training(mixsam,mixfra,random_seed,inputArgs.outputDir,num_epoches,lr,batchsize,ncuda)
        pd.DataFrame(list(mixfra.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_celltypes.txt',sep='\t')
        pd.DataFrame(list(mixsam.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_feature.txt',sep='\t')
    if (inputArgs.subcommand=='prediction'):  
        ncuda = 0
        test_sample = pd.read_csv(inputArgs.inputExp, sep="\t", index_col=0)
        feature = pd.read_csv(inputArgs.feature,sep='\t')['0']
        celltypes = pd.read_csv(inputArgs.cellType,sep='\t')['0']
        
        model = prediction.model_load(feature, celltypes, inputArgs.model, inputArgs.outputDir, random_seed,ncuda)
        result = prediction.dnn_prediction(model, test_sample, celltypes, feature,inputArgs.outputDir,ncuda)
        result.to_csv(inputArgs.outputDir+'/DAISM-DNN_result.txt',sep='\t')
if __name__ == "__main__":
    main()


