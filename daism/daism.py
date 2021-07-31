#!/usr/bin/env python
import os,sys
import numpy as np
import pandas as pd
import argparse

daismdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,daismdir)

import daism.modules.simulation as simulation
import daism.modules.training as training
import daism.modules.prediction as prediction
#--------------------------------------        
#--------------------------------------        

# main()

parser = argparse.ArgumentParser(description='DAISM-XMBD deconvolution.')
subparsers = parser.add_subparsers(dest='subcommand', help='Select one of the following sub-commands')

# create the parser for the "one-stop DAISM-DNN" command
parser_a = subparsers.add_parser('DAISM', help='one-stop DAISM-XMBD',description="one-stop DAISM-XMBD")
#parser_a.add_argument("-cell", type=str, help="The mode of cell types, [C]: Coarse, [F]: Fine", default='C')
parser_a.add_argument("-platform", type=str, help="Platform of calibration data, [R]: RNA-seq TPM, [S]: single cell RNA-seq", default="S")
parser_a.add_argument("-caliexp", type=str, help="Calibration samples expression file", default=None)
parser_a.add_argument("-califra", type=str, help="Calibration samples ground truth file", default=None)
parser_a.add_argument("-aug", type=str, help="Purified samples expression (h5ad)", default=None)
parser_a.add_argument("-N", type=int, help="Simulation samples number", default=16000)
parser_a.add_argument("-testexp", type=str, help="Test samples expression file", default=None)
parser_a.add_argument("-net", type=str, help="Network architecture used for training", default="coarse")
parser_a.add_argument("-outdir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "DAISM simulation" command
parser_b = subparsers.add_parser('DAISM_simulation', help='training set simulation using DAISM strategy',description='training set simulation using DAISM strategy.')
parser_b.add_argument("-platform", type=str, help="Platform of calibration data, [R]: RNA-seq TPM, [S]: single cell RNA-seq", default="S")
parser_b.add_argument("-caliexp", type=str, help="Calibration samples expression file", default=None)
parser_b.add_argument("-califra", type=str, help="Calibration samples ground truth file", default=None)
parser_b.add_argument("-aug", type=str, help="Purified samples expression (h5ad)", default=None)
parser_b.add_argument("-testexp", type=str, help="Test samples expression file", default=None)
parser_b.add_argument("-N", type=int, help="Simulation samples number", default=16000)
parser_b.add_argument("-outdir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "Generic simulation" command
parser_c = subparsers.add_parser('Generic_simulation', help='training set simulation using purified cells only',description='training set simulation using purified cells only.')
parser_c.add_argument("-platform", type=str, help="Platform of calibration data, [R]: RNA-seq TPM, [S]: single cell RNA-seq", default="S")
parser_c.add_argument("-aug", type=str, help="Purified samples expression (h5ad)", default=None)
parser_c.add_argument("-testexp", type=str, help="Test samples expression file", default=None)
parser_c.add_argument("-N", type=int, help="Simulation samples number", default=16000)
parser_c.add_argument("-outdir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "training" command
parser_d = subparsers.add_parser('training', help='train DNN model',description='train DNN model.')
parser_d.add_argument("-trainexp", type=str, help="Simulated samples expression file", default=None)
parser_d.add_argument("-trainfra", type=str, help="Simulated samples ground truth file", default=None)
parser_d.add_argument("-net", type=str, help="Network architecture used for training", default="coarse")
parser_d.add_argument("-outdir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "prediction" command
parser_e = subparsers.add_parser('prediction', help='predict using a trained model',description='predict using a trained model.')
parser_e.add_argument("-testexp", type=str, help="Test samples expression file", default=None)
parser_e.add_argument("-model", type=str, help="Deep-learing model file trained by DAISM", default="../output/DAISM_model.pkl")
parser_e.add_argument("-celltype", type=str, help="Model celltypes", default="../output/DAISM_model_celltypes.txt")
parser_e.add_argument("-feature", type=str, help="Model feature", default="../output/DAISM_model_feature.txt")
parser_e.add_argument("-net", type=str, help="Network architecture used for training", default="coarse")
parser_e.add_argument("-outdir", type=str, help="Output result file directory", default="../output/")


class Options:
    random_seed = 777
    min_f = 0.01
    max_f = 0.99
    lr = 1e-4
    batchsize = 64
    num_epoches = 500
    ncuda = 0


def main():
    # parse some argument lists
    inputArgs = parser.parse_args()

    if os.path.exists(inputArgs.outdir)==False:
        os.mkdir(inputArgs.outdir)


    #### DAISM modules ####

    if (inputArgs.subcommand=='DAISM'):

        # Load calibration data
        caliexp = pd.read_csv(inputArgs.caliexp, sep="\t", index_col=0)
        califra = pd.read_csv(inputArgs.califra, sep="\t", index_col=0)

        # Load test data
        test_sample = pd.read_csv(inputArgs.testexp, sep="\t", index_col=0)

        # Preprocess purified data
        mode = "daism"
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.aug,inputArgs.platform,mode,test_sample,caliexp,califra)

        # Create training dataset
        mixsam, mixfra, celltypes, feature = simulation.daism_simulation(caliexp,califra,C_all,Options.random_seed,inputArgs.N,inputArgs.platform,Options.min_f,Options.max_f)

        # Save signature genes and celltype labels
        pd.DataFrame(feature).to_csv(inputArgs.outdir+'/DAISM_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outdir+'/DAISM_celltypes.txt',sep='\t')

        print('Writing training data...')
        # Save training data
        mixsam.to_csv(inputArgs.outdir+'/DAISM_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outdir+'/DAISM_mixfra.txt',sep='\t')
        
        # Training model
        model = training.dnn_training(mixsam,mixfra,Options.random_seed,inputArgs.outdir,Options.num_epoches,Options.lr,Options.batchsize,Options.ncuda,inputArgs.net)

        # Save signature genes and celltype labels
        pd.DataFrame(list(mixfra.index)).to_csv(inputArgs.outdir+'/DAISM_model_celltypes.txt',sep='\t')
        pd.DataFrame(list(mixsam.index)).to_csv(inputArgs.outdir+'/DAISM_model_feature.txt',sep='\t')

        # Prediction
        result = prediction.dnn_prediction(model, test_sample, list(mixfra.index), list(mixsam.index),Options.ncuda,inputArgs.net)

        # Save predicted result
        result.to_csv(inputArgs.outdir+'/DAISM_result.txt',sep='\t')
    
    ############################
    #### simulation modules ####
    ############################

    #### DAISM simulation modules ####

    if (inputArgs.subcommand=='DAISM_simulation'):

        # Load calibration data
        caliexp = pd.read_csv(inputArgs.caliexp, sep="\t", index_col=0)
        califra = pd.read_csv(inputArgs.califra, sep="\t", index_col=0)

        # Load test data
        test_sample = pd.read_csv(inputArgs.testexp, sep="\t", index_col=0)

        # Preprocess purified data
        mode ="daism"
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.aug,inputArgs.platform,mode,test_sample,caliexp,califra)

        # Create training dataset
        mixsam, mixfra, celltypes, feature = simulation.daism_simulation(caliexp,califra,C_all,Options.random_seed,inputArgs.N,inputArgs.platform,Options.min_f,Options.max_f)
            
        # Save signature genes and celltype labels
        pd.DataFrame(feature).to_csv(inputArgs.outdir+'/DAISM_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outdir+'/DAISM_celltypes.txt',sep='\t')

        print('Writing training data...')
        # Save training data
        mixsam.to_csv(inputArgs.outdir+'/DAISM_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outdir+'/DAISM_mixfra.txt',sep='\t')
    
    #### Generic simulation modules ####

    if (inputArgs.subcommand=='Generic_simulation'):

        # Load test data
        test_sample = pd.read_csv(inputArgs.testexp, sep="\t", index_col=0)

        # Preprocess purified data
        mode = "generic"
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.aug,inputArgs.platform,mode,test_sample)

        # Create training dataset
        mixsam, mixfra, celltypes, feature = simulation.generic_simulation(C_all,Options.random_seed,inputArgs.N,inputArgs.platform,commongenes)
            
        # Save signature genes and celltype labels
        pd.DataFrame(feature).to_csv(inputArgs.outdir+'/Generic_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outdir+'/Generic_celltypes.txt',sep='\t')

        print('Writing training data...')
        # Save training data
        mixsam.to_csv(inputArgs.outdir+'/Generic_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outdir+'/Generic_mixfra.txt',sep='\t')


    ##########################
    #### training modules ####
    ##########################

    if (inputArgs.subcommand=='training'):
        # Load training data
        mixsam = pd.read_csv(inputArgs.trainexp, sep="\t", index_col=0)
        mixfra = pd.read_csv(inputArgs.trainfra, sep="\t", index_col=0)

        # Training model
        model = training.dnn_training(mixsam,mixfra,Options.random_seed,inputArgs.outdir,Options.num_epoches,Options.lr,Options.batchsize,Options.ncuda,inputArgs.net)

        # Save signature genes and celltype labels
        pd.DataFrame(list(mixfra.index)).to_csv(inputArgs.outdir+'/DAISM_model_celltypes.txt',sep='\t')
        pd.DataFrame(list(mixsam.index)).to_csv(inputArgs.outdir+'/DAISM_model_feature.txt',sep='\t')

    ############################
    #### prediction modules ####
    ############################

    if (inputArgs.subcommand=='prediction'):  
        # Load test data
        test_sample = pd.read_csv(inputArgs.testexp, sep="\t", index_col=0)

        # Load signature genes and celltype labels
        feature = pd.read_csv(inputArgs.feature,sep='\t')['0']
        celltypes = pd.read_csv(inputArgs.celltype,sep='\t')['0']
        
        # Load trained model
        model = prediction.model_load(feature, celltypes, inputArgs.model, Options.random_seed,Options.ncuda,inputArgs.net)

        # Prediction
        result = prediction.dnn_prediction(model, test_sample, celltypes, feature,Options.ncuda)

        # Save predicted result
        result.to_csv(inputArgs.outdir+'/DAISM_result.txt',sep='\t')


if __name__ == "__main__":
    main()


