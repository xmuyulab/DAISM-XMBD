######################################
##  Prediction module of DAISM-XMBD  ##
######################################

import os,sys
import torch
import pandas as pd
from torch.autograd import Variable

daismdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,daismdir)

from daism.modules.training import MLP,minmaxscaler


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

def model_load(commongene, celltypes, modelpath, random_seed, ncuda):
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
    model = MLP(INPUT_SIZE = len(commongene),OUTPUT_SIZE = len(celltypes)).double()

    # Load trained model
    if torch.cuda.is_available():
        model = model.cuda(ncuda)
        model.load_state_dict(torch.load(modelpath))
    else:
        model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        
    return model