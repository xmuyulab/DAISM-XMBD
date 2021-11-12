##########################
##  Training DAISM-XMBD  ##
##########################

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def minmaxscaler(x):
    """
    Data scaling using log-min-max
    :param x:
    :return:
    """

    x = np.log2(x + 1)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    return x

def Dataloader(xtr,ytr,batchsize):
    """
    Sample minibatches from a dataset for Pytorch
    :param xtr: training data (expression)
    :param ytr: training data (fraction)
    :param batchsize:
    :return:
    """

    train_dataset = Data.TensorDataset(xtr, ytr)
    trainloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size = batchsize,
        shuffle=True,
        num_workers=0    # set multi-work num read data
    )

    return trainloader

class train_preprocessing():
    """
    Split datasets into training part and validaton part, and turn it into the desired format
    """
    def __init__(self, tx, ty, ts, rs, ncuda):

        tx = minmaxscaler(tx)
        ty = ty.values
        xtr, xve, ytr, yve = train_test_split(tx.T, ty.T, test_size=ts, random_state=rs)

        self.xtr = torch.from_numpy(xtr)
        self.xve = torch.from_numpy(xve)
        self.ytr = torch.from_numpy(ytr)
        self.yve = torch.from_numpy(yve)
        
        if torch.cuda.is_available():
            self.xve = self.xve.cuda(ncuda) 
            self.yve = self.yve.cuda(ncuda) 
            

class MLP(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP, self).__init__() 
        # Architectures 
        L1 = 1024
        L2 = 512
        L3 = 256
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.ReLU(),
            nn.Linear(L1,L2),
            nn.BatchNorm1d(L2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.ReLU(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L3, OUTPUT_SIZE),
        )
    def forward(self, x):   
        y = self.hidden(x)    
        y = self.predict(y)   
        return y

def evaluate(model,xve,yve):
    """
    In each epoch, use validation data to evaluate model
    :param model:
    :param xve: validation data (expression)
    :param yve: validation data (fraction)
    :return: mean absolute error of validation data
    """

    model.eval()
    vout = model(xve)
    ve_p = Variable(vout,requires_grad = False).cpu().numpy().reshape(yve.shape[0]*yve.shape[1])
    ve_y = Variable(yve,requires_grad = False).cpu().numpy().reshape(yve.shape[0]*yve.shape[1])
    res = np.abs(ve_p-ve_y)
    mae_ve = np.mean(res)

    return mae_ve

def dnn_training(mixsam,mixfra,random_seed,modelpath,num_epoches=300,lr=1e-4,batchsize=64,ncuda=0):

    print('Model training start!')

    # Fixed parameter definition
    lr_min = 1e-5   # Minimum learning rate
    de_lr = 0.9    # Attenuation index
    mae_tr_prev = 0.05   # training mae initial value
    dm = 0   # Attenuation threshold
    mae_ve = []
    min_mae = 1
    n = 0
    min_epoch = 20 
    cn = mixfra.shape[0]
    gn = mixsam.shape[0]

    # Data preprocessing
    data = train_preprocessing(tx = mixsam,ty = mixfra, ts = 0.2, rs = random_seed, ncuda = ncuda)
    trainloader = Dataloader(xtr=data.xtr, ytr=data.ytr, batchsize=batchsize)
    
    # Model definition
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    model = MLP(INPUT_SIZE = gn,OUTPUT_SIZE = cn).double()

    if torch.cuda.is_available():
        model = model.cuda(ncuda)  

    optimizer = torch.optim.Adam(model.parameters(), lr= lr)  
    loss_func = torch.nn.MSELoss()     
    
    # Training loop
    for epoch in range(num_epoches):
        
        mae_tr=[]
        for step, (batch_x, batch_y) in enumerate(trainloader):

            batch_x = batch_x.cuda(ncuda)
            batch_y = batch_y.cuda(ncuda)
            model.train()
            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_func(out, batch_y) 
            loss.backward() 
            optimizer.step() 
            tr_p = Variable(out,requires_grad = False).cpu().numpy().reshape(batchsize*cn)  
            tr_t = Variable(batch_y,requires_grad = False).cpu().numpy().reshape(batchsize*cn)
            mae_tr.append(np.mean(abs(tr_p - tr_t)))
        
        # learning rate decay
        mae_tr_change = (np.mean(mae_tr)-mae_tr_prev)
        mae_tr_prev = np.mean(mae_tr)
        if mae_tr_change > dm:         
            optimizer.param_groups[0]['lr'] *= de_lr   
        if optimizer.param_groups[0]['lr'] < lr_min:
            optimizer.param_groups[0]['lr'] = lr_min

        # early-stopping
        mae_ve.append(evaluate(model,data.xve,data.yve))        
        if epoch >= min_epoch:
            if mae_ve[epoch] <= min_mae:
                min_mae = mae_ve[epoch]
                torch.save(model.state_dict(), modelpath+'DAISM_model.pkl')
                n = 0
            else:
                n += 1
            if n==10:
                break

    model.load_state_dict(torch.load(modelpath+'DAISM_model.pkl'))

    print("Model training finish!")

    return model