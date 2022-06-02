######################################
##  Prediction module of DAISM-XMBD  ##
######################################

import os,sys
import torch
import pandas as pd
from torch.autograd import Variable
from math import sqrt
from sklearn.metrics import mean_squared_error

daismdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,daismdir)

from daism.modules.training import MLP,MLP_sum2one,minmaxscaler


def dnn_prediction(model, testsam, celltypes, feature,ncuda):
    print("Result prediction start!")
    # preprocess test data
    data = testsam.reindex(feature)   
    data = minmaxscaler(data).T
    data = torch.from_numpy(data)
    if torch.cuda.is_available():
        data = data.cuda(ncuda)
        
    model.eval()
    out = model(data)
    
    pred = Variable(out,requires_grad=False).cpu().numpy().reshape(testsam.shape[1],len(celltypes))
    pred[pred<0]=0    
    
    pred_result = pd.DataFrame(pred.T,index=celltypes,columns=testsam.columns)
    
    print("Result prediction finish!")
    return pred_result

def model_load(commongene, celltypes, modelpath, random_seed, ncuda,sum2one = False):
    """
    Load trained model
    :param commongene:
    :param celltypes:
    :param modelpath:
    :param outdir:
    :param random_seed:
    :param ncuda:
    :return:
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Initialize model
    if sum2one == True:
        model = MLP_sum2one(INPUT_SIZE = len(commongene),OUTPUT_SIZE = len(celltypes)).double()
    else:
        model = MLP(INPUT_SIZE = len(commongene),OUTPUT_SIZE = len(celltypes)).double()

    # Load trained model
    if torch.cuda.is_available():
        model = model.cuda(ncuda)
        model.load_state_dict(torch.load(modelpath))
    else:
        model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        
    return model

def ccc(y,p):
    y_mean = y.mean()
    p_mean = p.mean()
    md = sum((y-p)**2)/len(y)
    y_var = sum((y-y_mean)**2)/len(y)
    p_var = sum((p-p_mean)**2)/len(p)
    bottom = y_var+p_var+(y_mean-p_mean)**2
    return 1-md/bottom


def do_eval(result,ground_truth):

    result_corr = lambda x : x["prediction"].corr(x["measured"])
    result_spearman = lambda x : x["prediction"].corr(x["measured"],'spearman')
    result_ccc = lambda x : ccc(x["measured"],x["prediction"])
    result_rmse = lambda x : sqrt(mean_squared_error(x["prediction"], x["measured"]))

    result['cell.type']=result.index
    result_melt = pd.melt(result,id_vars='cell.type',var_name='sample.id',value_name='prediction')
    ground_truth['cell.type']=ground_truth.index
    gt_melt = pd.melt(ground_truth,id_vars='cell.type',var_name='sample.id',value_name='measured')
    final_result = pd.merge(result_melt,gt_melt,on=['cell.type','sample.id'])
    by_cell_type = final_result.groupby(["cell.type"])

    corr_result= pd.DataFrame({'pearson':by_cell_type.apply(result_corr),
                          'CCC':by_cell_type.apply(result_ccc),
                          'spearman':by_cell_type.apply(result_spearman),
                          'RMSE':by_cell_type.apply(result_rmse)}).T
    corr_result['mean'] = corr_result.apply(lambda x: x.mean(), axis=1)

    return corr_result