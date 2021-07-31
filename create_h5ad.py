##############################
## cread purified h5ad file ##
##############################

# input: annotation table and the whole expression profile
# output: purified h5ad file

import os
import pandas as pd
import anndata
import argparse
import gc
import numpy as np

parser = argparse.ArgumentParser(description='cread purified h5ad file for DAISM-XMBD')
parser.add_argument("-anno", type=str, help="annotation table (contains 'sample.name' and 'cell.type' two columns)", default=None)
parser.add_argument("-exp", type=str, help="the whole expression profile (sample.name in column and gene symbol in row)", default=None)
parser.add_argument("-outdir", type=str, help="the directory to store h5ad file", default="example/")
parser.add_argument("-prefix",type=str,help="the prefix of h5ad file",default= "purified")

def main():
    inputArgs = parser.parse_args()

    if os.path.exists(inputArgs.outdir)==False:
        os.mkdir(inputArgs.outdir)

    anno_table = pd.read_csv(inputArgs.anno)

    cell_list = list(anno_table['cell.type'].unique())

    exp = pd.read_csv(inputArgs.exp,sep="\t",index_col=0)

    adata = []
    for cell in cell_list:
        tmp = anno_table[anno_table['cell.type']==cell]
        sample_list = tmp['sample.name']
        sample_list_inter = list(set(sample_list).intersection(list(exp.columns)))
        exp_select=exp[sample_list_inter]

        anno = pd.DataFrame(np.repeat(cell,exp_select.shape[1]),columns=['cell.type'])
    
        adata.append(anndata.AnnData(X=exp_select.T.values,
                            obs=anno,
                            var=pd.DataFrame(columns=[],index=list(exp_select.index))))
  
    for i in range(1, len(adata)):
        print("Concatenating " + str(i))
        adata[0] = adata[0].concatenate(adata[1])
        del adata[1]
        gc.collect()
        print(len(adata))

    adata = adata[0]

    adata.write(inputArgs.outdir+'/'+inputArgs.prefix+'.h5ad')

if __name__ == "__main__":
    main()