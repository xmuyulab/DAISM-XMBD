#!/usr/bin/env python
import os,sys
import numpy as np
import pandas as pd
import argparse

daismdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,daismdir)

import daism_dnn.modules.simulation as simulation
import daism_dnn.modules.training as training
import daism_dnn.modules.prediction as prediction
#--------------------------------------        
#--------------------------------------        

# main()

parser = argparse.ArgumentParser(description='DAISM-DNN deconvolution.')
subparsers = parser.add_subparsers(dest='subcommand', help='Select one of the following sub-commands')

# create the parser for the "one-stop DAISM-DNN" command
parser_a = subparsers.add_parser('DAISM-DNN', help='DAISM-DNN')
#parser_a.add_argument("-cell", type=str, help="The mode of cell types, [C]: Coarse, [F]: Fine", default='C')
parser_a.add_argument("-platform", type=str, help="Platform of [calibration data] + [purified data for agumentation], [Rs]: RNA-seq TPM + scRNA, [Rt]: RNA-seq TPM + TPM, [Ms]: Microarray + scRNA", default="Rs")
parser_a.add_argument("-caliExp", type=str, help="Calibration samples expression file", default=None)
parser_a.add_argument("-caliFra", type=str, help="Calibration samples ground truth file", default=None)
parser_a.add_argument("-pureExp", type=str, help="Purified samples expression (h5ad)", default=None)
parser_a.add_argument("-simNum", type=int, help="Simulation samples number", default=16000)
parser_a.add_argument("-inputExp", type=str, help="Test samples expression file", default=None)
parser_a.add_argument("-outputDir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "simulation" command
parser_b = subparsers.add_parser('simulation', help='simulation')
parser_b.add_argument("-platform", type=str, help="Platform of [calibration data] + [purified data for agumentation], [Rs]: RNA-seq TPM + scRNA, [Rt]: RNA-seq TPM + TPM, [Ms]: Microarray + scRNA", default="Rs")
parser_b.add_argument("-caliExp", type=str, help="Calibration samples expression file", default=None)
parser_b.add_argument("-caliFra", type=str, help="Calibration samples ground truth file", default=None)
parser_b.add_argument("-pureExp", type=str, help="Purified samples expression (h5ad)", default=None)
parser_b.add_argument("-inputExp", type=str, help="Test samples expression file", default=None)
parser_b.add_argument("-simNum", type=int, help="Simulation samples number", default=16000)
parser_b.add_argument("-outputDir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "training" command
parser_c = subparsers.add_parser('training', help='training')
parser_c.add_argument("-trainExp", type=str, help="Simulated samples expression file", default=None)
parser_c.add_argument("-trainFra", type=str, help="Simulated samples ground truth file", default=None)
parser_c.add_argument("-outputDir", type=str, help="Output result file directory", default="../output/")

# create the parser for the "prediction" command
parser_d = subparsers.add_parser('prediction', help='prediction')
parser_d.add_argument("-inputExp", type=str, help="Test samples expression file", default=None)
parser_d.add_argument("-model", type=str, help="Deep-learing model file trained by DAISM-DNN", default="../output/dnn_daism_model.pkl")
parser_d.add_argument("-cellType", type=str, help="Model celltypes", default="../output/dnn_daism_celltypes.txt")
parser_d.add_argument("-feature", type=str, help="Model feature", default="../output/dnn_daism_feature.txt")
parser_d.add_argument("-outputDir", type=str, help="Output result file directory", default="../output/")

class Options:
    random_seed = 777
    min_f = 0.01
    max_f = 0.99
    lr = 1e-4
    batchsize = 32
    num_epoches = 500
    ncuda = 0


def main():
    # parse some argument lists
    inputArgs = parser.parse_args()

    if os.path.exists(inputArgs.outputDir)==False:
        os.mkdir(inputArgs.outputDir)


    #### DAISM-DNN modules ####

    if (inputArgs.subcommand=='DAISM-DNN'):
        # Load calibration data
        if inputArgs.caliExp == None and inputArgs.caliFra == None:
            mode = "puremix"
            caliexp = None
            califra = None
        else:
            mode = "daism"
            caliexp = pd.read_csv(inputArgs.caliExp, sep="\t", index_col=0)
            califra = pd.read_csv(inputArgs.caliFra, sep="\t", index_col=0)

        # Load test data
        test_sample = pd.read_csv(inputArgs.inputExp, sep="\t", index_col=0)

        # Preprocess purified data
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.pureExp,inputArgs.platform,mode,test_sample,caliexp,califra)

        # Create training dataset
        if mode == "daism":
            mixsam, mixfra, celltypes, feature = simulation.daism_simulation(caliexp,califra,C_all,Options.random_seed,inputArgs.simNum,inputArgs.platform,commongenes,Options.min_f,Options.max_f)
        if mode == "puremix":
            mixsam, mixfra, celltypes, feature = simulation.puremix_simulation(C_all,Options.random_seed,inputArgs.simNum,inputArgs.platform,commongenes)

        # Save signature genes and celltype labels
        pd.DataFrame(feature).to_csv(inputArgs.outputDir+'/DAISM-DNN_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outputDir+'/DAISM-DNN_celltypes.txt',sep='\t')

        print('Writing training data...')
        # Save training data
        mixsam.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixfra.txt',sep='\t')
        
        # Training model
        model = training.dnn_training(mixsam,mixfra,Options.random_seed,inputArgs.outputDir,Options.num_epoches,Options.lr,Options.batchsize,Options.ncuda)

        # Save signature genes and celltype labels
        pd.DataFrame(list(mixfra.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_celltypes.txt',sep='\t')
        pd.DataFrame(list(mixsam.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_feature.txt',sep='\t')

        # Prediction
        result = prediction.dnn_prediction(model, test_sample, list(mixfra.index), list(mixsam.index),inputArgs.outputDir,Options.ncuda)

        # Save predicted result
        result.to_csv(inputArgs.outputDir+'/DAISM-DNN_result.txt',sep='\t')
    

    #### simulation modules ####

    if (inputArgs.subcommand=='simulation'):
        # Load calibration data
        if inputArgs.caliExp == None and inputArgs.caliFra == None:
            mode = "puremix"
            caliexp = None
            califra = None
        else:
            mode = "daism"
            caliexp = pd.read_csv(inputArgs.caliExp, sep="\t", index_col=0)
            califra = pd.read_csv(inputArgs.caliFra, sep="\t", index_col=0)

        # Load test data
        test_sample = pd.read_csv(inputArgs.inputExp, sep="\t", index_col=0)

        # Preprocess purified data
        commongenes,caliexp,C_all = simulation.preprocess_purified(inputArgs.pureExp,inputArgs.platform,mode,test_sample,caliexp,califra)

        # Create training dataset
        if mode == "daism":
            mixsam, mixfra, celltypes, feature = simulation.daism_simulation(caliexp,califra,C_all,Options.random_seed,inputArgs.simNum,inputArgs.platform,commongenes,Options.min_f,Options.max_f)
        if mode == "puremix":
            mixsam, mixfra, celltypes, feature = simulation.puremix_simulation(C_all,Options.random_seed,inputArgs.simNum,inputArgs.platform,commongenes)
            
        # Save signature genes and celltype labels
        pd.DataFrame(feature).to_csv(inputArgs.outputDir+'/DAISM-DNN_feature.txt',sep='\t')
        pd.DataFrame(celltypes).to_csv(inputArgs.outputDir+'/DAISM-DNN_celltypes.txt',sep='\t')

        print('Writing training data...')
        # Save training data
        mixsam.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixsam.txt',sep='\t')
        mixfra.to_csv(inputArgs.outputDir+'/DAISM-DNN_mixfra.txt',sep='\t')
    

    #### training modules ####

    if (inputArgs.subcommand=='training'):
        # Load training data
        mixsam = pd.read_csv(inputArgs.trainExp, sep="\t", index_col=0)
        mixfra = pd.read_csv(inputArgs.trainFra, sep="\t", index_col=0)

        # Training model
        model = training.dnn_training(mixsam,mixfra,random_seed,inputArgs.outputDir,Options.num_epoches,Options.lr,Options.batchsize,Options.ncuda)

        # Save signature genes and celltype labels
        pd.DataFrame(list(mixfra.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_celltypes.txt',sep='\t')
        pd.DataFrame(list(mixsam.index)).to_csv(inputArgs.outputDir+'/DAISM-DNN_model_feature.txt',sep='\t')


    #### prediction modules ####

    if (inputArgs.subcommand=='prediction'):  
        # Load test data
        test_sample = pd.read_csv(inputArgs.inputExp, sep="\t", index_col=0)

        # Load signature genes and celltype labels
        feature = pd.read_csv(inputArgs.feature,sep='\t')['0']
        celltypes = pd.read_csv(inputArgs.cellType,sep='\t')['0']
        
        # Load trained model
        model = prediction.model_load(feature, celltypes, inputArgs.model, inputArgs.outputDir, Options.random_seed,Options.ncuda)

        # Prediction
        result = prediction.dnn_prediction(model, test_sample, celltypes, feature,inputArgs.outputDir,Options.ncuda)

        # Save predicted result
        result.to_csv(inputArgs.outputDir+'/DAISM-DNN_result.txt',sep='\t')


if __name__ == "__main__":
    main()


