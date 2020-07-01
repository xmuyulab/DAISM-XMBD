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
import scanpy.api as sc
import anndata
import math

# test function
# select gene
def sc_norm(exp,counts_per_cell_after):
    adata = anndata.AnnData(exp)
    sc.pp.normalize_per_cell(adata,counts_per_cell_after=counts_per_cell_after)
    return np.array(adata.X)
#进度条显示
class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' +'('+'%d' % self.i+'/'+'%d' %self.max_steps+')'+ '\r' 
        #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

def minmaxscaler(x):
    x = np.log2(x + 1)
    x = (x - x.min(axis = 0))/(x.max(axis = 0) - x.min(axis = 0))
    return x

def Dataloader(xtr,ytr,batchsize):
    """
    :ytr - fraction cell type*sample

    :batchsize 
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
    def __init__(self, tx, ty, ts, rs, ncuda):

        tx = minmaxscaler(tx).values
        ty = ty.values
        xtr, xve, ytr, yve = train_test_split(tx.T, ty.T, test_size=ts, random_state=rs)

        self.xtr = torch.from_numpy(xtr)
        self.xve = torch.from_numpy(xve)
        self.ytr = torch.from_numpy(ytr)
        self.yve = torch.from_numpy(yve)
        
        if torch.cuda.is_available():
            self.xtr = self.xtr.cuda(ncuda)
            self.xve = self.xve.cuda(ncuda) 
            self.ytr = self.ytr.cuda(ncuda)
            self.yve = self.yve.cuda(ncuda) 
            
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
class MLP0(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP0, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        L1 = 512
        L2 = 1024
        L3 = 512
        L4 = 128
        L5 = 32
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            # nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            #nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(L3,L4),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
            # nn.Softmax()
        )
    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        y = self.hidden(x)     # 激励函数(隐藏层的线性值)
        y = self.predict(y)    # 输出值
        return y    
def evaluate(model,xve,yve,epoch):

    model.eval()
    vout = model(xve)
    ve_p = Variable(vout,requires_grad = False).cpu().numpy().reshape(yve.shape[0]*yve.shape[1])
    ve_y = Variable(yve,requires_grad = False).cpu().numpy().reshape(yve.shape[0]*yve.shape[1])
    res = np.abs(ve_p-ve_y)
    mae_ve = np.mean(res)
    return mae_ve

def dnn_training(mixsam,mixfra,random_seed,outdir,modelpath,num_epoches=300,lr=1e-4,batchsize=64,ncuda=0):
    
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
    
    # training
    for epoch in range(num_epoches):
        
        mae_tr=[]
        for step, (batch_x, batch_y) in enumerate(trainloader):

            model.train()
            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_func(out, batch_y) 
            loss.backward() 
            optimizer.step() 
            tr_p = Variable(out,requires_grad = False).cpu().numpy().reshape(batchsize*cn)  
            tr_t = Variable(batch_y,requires_grad = False).cpu().numpy().reshape(batchsize*cn)
            mae_tr.append(np.mean(abs(tr_p - tr_t)))
        
        #print('Epoch {}/{},MSEloss:{:.4f}'.format(epoch, num_epoches, loss.item()))
        mae_tr_change = (np.mean(mae_tr)-mae_tr_prev)
        mae_tr_prev = np.mean(mae_tr)
        if mae_tr_change > dm:         
            optimizer.param_groups[0]['lr'] *= de_lr   
        if optimizer.param_groups[0]['lr'] < lr_min:
            optimizer.param_groups[0]['lr'] = lr_min

        mae_ve.append(evaluate(model,data.xve,data.yve,epoch))        
        if epoch >= min_epoch:
            if mae_ve[epoch] <= min_mae:
                min_mae = mae_ve[epoch]
                torch.save(model.state_dict(), modelpath+'dnn_daism_model1.pkl')
                n = 0
            else:
                n += 1
            if n==10:
                break

    model.load_state_dict(torch.load(modelpath+'dnn_daism_model1.pkl'))
    print("model_trainging finish!")
    return model

def dnn_prediction(model, testsam, celltypes, feature,outdir,ncuda):

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

def get_celltype_list(datapath , suffix):
    """
    datapath: the directory of purified data
    suffix: the suffix of purified data file
    """
    
    file_list = os.listdir(datapath)
    
    cell_list = []
    for i,file in enumerate(file_list):
        cell = file.split(suffix)[0]
        cell_list.append(cell)
        
    return(cell_list)#返回list

def preprocess_purified(datapath,suffix,platform,samexp):
    """
    
    """
    
    cell_list = get_celltype_list(datapath, ".txt")
    #print(cell_list)
    C_df = {}#pandas dict
    C_all = {} #numpy dict
    
    #for Microarray
    if platform == "Ma":
        #cell_list = get_celltype_list(datapath , suffix)
        for cell in cell_list:
            C_df[cell] = pd.read_csv(datapath+cell+suffix, index_col=0,sep='\t')           

        if cancer_path != None:
            C_df['cancer'] = pd.read_csv(cancer_path, index_col=0,sep='\t')  
            commongenes = list(set(list(C_df['cancer'].index)).intersection(list(C_df[cell_list[0]].index)))
            commongenes.sort()
            C_all['cancer'] = C_df['cancer'].reindex(commongenes).T.values
        else:
            commongenes = C_df[cell_list[0]].index

        for cell in cell_list:
            C_all[cell] = C_df[cell].reindex(commongenes).T.values
            C_all[cell] = 2**C_all[cell]
    
    #for RNA-seq
    elif platform =="Rc" or platform =="Rt":
    
        for cell in cell_list:
            C_df[cell] = pd.read_csv(datapath+'/'+cell+suffix, index_col=0,sep='\t')           

        commongenes = list(set(list(samexp.index)).intersection(list(C_df[cell_list[0]].index)))
        samexp = samexp.reindex(commongenes)
        
        for cell in cell_list:
            C_all[cell] = C_df[cell].reindex(commongenes).T.values
        
    #for scRNA-seq
    elif platform == "Rs" or platform == "Ms":
        for cell in cell_list:
            C_df[cell] = pd.read_csv(datapath+'/'+cell+suffix, index_col=0,sep='\t')
            
        
        commongenes = list(set(list(samexp.index)).intersection(list(C_df[cell_list[0]].index)))
        samexp = samexp.reindex(commongenes)
        
        for cell in cell_list:
            C_all[cell] = C_df[cell].reindex(commongenes).T.values
            
        counts_per_cell_after = np.median(np.vstack(tuple(C_all.values())).sum(axis=1))

        for cell in cell_list:
            C_all[cell] = sc_norm(C_all[cell],counts_per_cell_after=counts_per_cell_after)
    
    return(commongenes,samexp,C_all) 

def daism_simulation(trainexp, trainfra,C_all, random_seed, N, outdir,platform, mode,marker,min_f=0.01, max_f=0.99,sparse=False):
    gn = trainexp.shape[0]
    cn = trainfra.shape[0]
    sn = trainfra.shape[1]

    if N % sn == 0:
        n = int(N/sn)
    else:
        n = int(N/sn)+1
        
    random.seed(random_seed)
    np.random.seed(random_seed)

    process_bar = ShowProcess(N, 'mixture_simulation finish!')

    mixsam = np.zeros(shape=(N, gn))
    mixfra = np.zeros(shape=(N, cn))
    
    for i,sampleid in enumerate(trainfra.columns):

        for j in range(n):
            if i*n+j < N:
                if platform == "Rc" or platform == "Rt" or platform == "Ma":
                    
                    random.seed(random_seed+i*n+j)
                    np.random.seed(random_seed+i*n+j)
                    
                    if mode == "Dc" or mode == "Df":
                        mix_fraction = round(random.uniform(min_f,max_f),8)
                    elif mode == "Pc" or mode == "Pf":
                        mix_fraction = 0 
                    
                    if sparse:
                        no_keep = np.random.randint(1, cn)
                        keep = np.random.choice(list(range(cn)), size=no_keep, replace=False)#看是哪几个留下
                        fraction_sp = np.random.dirichlet([1]*no_keep, 1)*(1-mix_fraction)
                        fraction = np.array([0]*cn, ndmin =  2)
                        for m, act in enumerate(keep):
                            fraction[0,act] = fraction_sp[0,m]
   
                    else:
                        fraction = np.random.dirichlet([1]*cn,1)*(1-mix_fraction)
                    
                    mixsam[i*n+j] = np.array(trainexp.T.values[i])*mix_fraction
                    mixfra[i*n+j] = trainfra.T.values[i]*mix_fraction

                    for k, celltype in enumerate(trainfra[sampleid].T.index):

                        pure = C_all[celltype][random.randint(0,C_all[celltype].shape[0]-1)]

                        mixsam[i*n+j] += np.array(pure)*fraction[0,k]
                        mixfra[i*n+j][k] += fraction[0,k]
                    mixsam[i*n+j] = 1e+6*mixsam[i*n+j]/np.sum(mixsam[i*n+j])

                elif platform == "Rs" or platform =='Ms':
                    random.seed(random_seed+i*n+j)
                    np.random.seed(random_seed+i*n+j)
                    cell_sum = 500
                    if mode == "Dc" or mode == "Df":
                        mix_fraction = round(random.uniform(min_f,max_f),8)
                    elif mode == "Pc" or mode == "Pf":
                        mix_fraction = 0  
                    if sparse:
                        no_keep = np.random.randint(1, cn)
                        keep = np.random.choice(list(range(cn)), size=no_keep, replace=False)#看是哪几个留下
                        fraction_sp = np.random.dirichlet([1]*no_keep, 1)*(1-mix_fraction)
                        fraction = np.array([0]*cn, ndmin =  2)
                        for m, act in enumerate(keep):
                            fraction[0,act] = fraction_sp[0,m]
   
                    else:
                        fraction = np.random.dirichlet([1]*cn,1)*(1-mix_fraction)
                                         
                    
                    #mixsam[i*n+j] = np.array(trainexp.T.values[i])*mix_fraction
                    mixfra[i*n+j] = trainfra.T.values[i]*mix_fraction
                    cell_pure = math.ceil(500*(1-mix_fraction))
                    mix_pure = [0]*gn
                    for k, celltype in enumerate(trainfra[sampleid].T.index):
                        
                        
                        #print("celltype",celltype)
                        #print("celltype.shape",celltype.shape)
                        pure = C_all[celltype][np.random.choice(range(C_all[celltype].shape[0]),math.ceil(fraction[0,k]*cell_sum))].sum(axis=0)
                        if k==0:
                            pure_sum = pure
                        else:
                            pure_sum += pure

                        mixfra[i*n+j][k] += fraction[0,k]

                    #mix = np.vstack((cb,cd4,cd8,cmono,cnk)).sum(axis=0)+mix_fraction*real_mixsam
                 
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

    #mixsam.to_csv(outdir+"dnn_daism_mixsam.txt",sep="\t")
    #mixfra.to_csv(outdir+"dnn_daism_mixfra.txt",sep="\t")
    #pd.DataFrame(celltypes).to_csv(outdir+"dnn_daism_celltypes.txt",sep="\t")
    
    return (mixsam, mixfra, celltypes, feature)

def general_prediction(mode,testexp,general_modelpath,feature,celltypes):
    if mode == "Gc":
        total_celltypes = ['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells','monocytic.lineage','neutrophils', 'fibroblasts','endothelial.cells']
        final_result = np.zeros(shape=(len(total_celltypes),testexp.shape[1]))
        model = MLP(INPUT_SIZE=len(feature),OUTPUT_SIZE=len(total_celltypes)).double()
    elif mode == "Gf":
        total_celltypes = ['naive.B.cells', 'memory.B.cells', 'naive.CD4.T.cells','memory.CD4.T.cells','regulatory.T.cells', 'naive.CD8.T.cells','memory.CD8.T.cells', 'NK.cells', 'monocytes','myeloid.dendritic.cells', 'macrophages', 'neutrophils', 'fibroblasts','endothelial.cells']
        final_result = np.zeros(shape=(len(total_celltypes),testexp.shape[1]))
        model = MLP0(INPUT_SIZE=len(feature),OUTPUT_SIZE=len(total_celltypes)).double()
                  
    data = testexp.reindex(feature)
    data = minmaxscaler(data).values.T
    data = torch.from_numpy(data)
    file_list = os.listdir(general_modelpath)
        
    for i,file in enumerate(file_list):
        model.load_state_dict(torch.load(general_modelpath+file,map_location='cpu'))
        model.eval()
        out = model(data)          
        pred = Variable(out,requires_grad=False).cpu().numpy().reshape(testexp.shape[1],len(total_celltypes))
        if i == 0:
            final_pred = pred.T
        else:
            final_pred += pred.T
        final_result = final_pred.T/len(file_list)
    pred_result = pd.DataFrame(final_result.T,index=total_celltypes,columns=testexp.columns)
    pred_result = pred_result.reindex(celltypes)
    #pred_result.to_csv(outdir+"dnn_daism_result.txt",sep="\t")
    print("result_prediction finish!")
    return pred_result
