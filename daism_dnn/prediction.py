#
import torch
import pandas as pd
from torch.autograd import Variable
from training import MLP,minmaxscaler

def dnn_prediction(model, testsam, celltypes, feature,outdir,ncuda):
    print("Result prediction start!")
    data = testsam.reindex(feature)    
    data = minmaxscaler(data).values.T
    data = torch.from_numpy(data)
    if torch.cuda.is_available():
        data = data.cuda(ncuda)
        
    model.eval()
    out = model(data)
    
    pred = Variable(out,requires_grad=False).cpu().numpy().reshape(testsam.shape[1],len(celltypes))    
    
    pred_result = pd.DataFrame(pred.T,index=celltypes,columns=testsam.columns)
    
    print("Result prediction finish!")
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