import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable

class MLP(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP, self).__init__()  
        L1 = 256
        L2 = 512
        L3 = 128
        L4 = 32
        L5 = 16
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.BatchNorm1d(L2),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.Tanh(),
            nn.Linear(L3,L4),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
            # nn.Softmax()
        )
    def forward(self, x):   
        y = self.hidden(x)    
        y = self.predict(y)   
        return y

def minmaxscaler(x):
    x = np.log2(x + 1)
    x = (x - x.min(axis = 0))/(x.max(axis = 0) - x.min(axis = 0))
    return x

def dnn_prediction(model, testsam, celltypes, feature,outdir,ncuda):
    print("result_prediction start!")
    data = testsam.reindex(feature)    
    data = minmaxscaler(data).values.T
    data = torch.from_numpy(data)
    if torch.cuda.is_available():
        data = data.cuda(ncuda)
        
    model.eval()
    out = model(data)
    
    pred = Variable(out,requires_grad=False).cpu().numpy().reshape(testsam.shape[1],len(celltypes))    
    
    pred_result = pd.DataFrame(pred.T,index=celltypes,columns=testsam.columns)
    
    #pred_result.to_csv(outdir+"dnn_daism_result.txt",sep="\t")
    print("result_prediction finish!")
    return pred_result

def model_load(commongene, celltypes, modelpath, outdir, random_seed, ncuda):
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    model = MLP(INPUT_SIZE = len(commongene),OUTPUT_SIZE = len(celltypes)).double()
    if torch.cuda.is_available():
        model = model.cuda(ncuda)
        model.load_state_dict(torch.load(modelpath))
    else:
        model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        
    return model