## DAISM-XMBD

We propose data augmentation through in silico mixing with deep neural networks (DAISM-XMBD) to achieve highly accurate and unbiased immune-cell proportion estimation from bulk  RNA sequencing (RNA-seq) data. Our method tackles the batch effect problem by creating a data-specific training dataset from a small subset of samples with ground truth cell proportions which is further augmented with publicly available RNA-seq data from purified cells or single-cell RNA-seq (scRNA-seq) data.

A pre-print describing the method is available at bioRxiv:
 [DAISM-XMBD: Highly accurate cell type proportion estimation within silico data augmentation and deep neural networks](https://www.biorxiv.org/content/10.1101/2020.03.26.009308v2)
 
### Installation
#### Pip
It is recommended to create a new conda environment:
```
conda create -n daism python=3.7
```
Activate this environment:
```
conda activate daism
```
And run the following command to install daism via pip:
```
pip install daism
```

The main dependencies list below:
```
python (v3.7.7)
torch (v1.5.1)
pandas (v1.0.5)
numpy (v1.18.1)
scikit-learn (v0.23.1)
argh (v0.26.2) 
anndata (v0.7.3)
scanpy (v1.5.1)
tqdm
```

#### Docker
We provide a docker image with DAISM-XMBD installed:
[https://hub.docker.com/r/zoelin1130/daism](https://hub.docker.com/r/zoelin1130/daism)

Pull the docker image:
```
docker pull zoelin1130/daism:1.0
```
Create a container (GPU):
```
docker run --gpus all -i -t --name run_daism -v example:/workspace/example/ zoelin1130/daism:1.0 /bin/bash
```
Create a container (CPU):
```
docker run -i -t --name run_daism -v example:/workspace/example/ zoelin1130/daism:1.0 /bin/bash
```
```run_daism```is your container name. It is strongly recommended to add -v parameter for implementing data and scripts mounting: mount the local volume ```example``` (from your machine) to ```/workspace/example/``` (to your container) instead of directly copy them into the container.

### Cell Types Supported
The example we provide contains the following cell types. The purified dataset for data augmentation can be downloaded from:[https://figshare.com/s/b5737bec1ab6e1502b5a](https://figshare.com/s/b5737bec1ab6e1502b5a)

pbmc8k.h5ad contains 5 cell types: B.cells, CD4.T.cells, CD8.T.cells, monocytic.lineage, NK.cells.

pbmc8k_fine.h5ad contains 11 cell types: naive.B.cells, memory.B.cells, naive.CD4.T.cells, memory.CD4.T.cells,naive.CD8.T.cells, memory.CD8.T.cells, regulatory.T.cells, monocytes, macrophages, myeloid.dendritic.cells, NK.cells.

RNA_TPM_coarse.h5ad contains 5 cell types: B.cells, CD4.T.cells, CD8.T.cells, monocytic.lineage, NK.cells.

Note: each cell type needs to be named according to above format.

DAISM-XMBD can support the prediction of any cell types, as long as calibration samples with ground truth and purified expression profiles of corresponding cell types are provided.

### Usage
In our example below, we set working directory to daism. Use -h to print out help information on DAISM-XMBD modules.
```
cd daism
python daism.py -h

# if you installed daism via pip, you can used following command to print out help information.
daism -h
```

DAISM-XMBD consists of four modules:

#### DAISM modules: 
```
daism DAISM -platform S -caliexp ../example/caliexp.txt -califra ../example/califra.txt -aug ../example/pbmc8k.h5ad -N 16000 -testexp ../example/testexp.txt -net coarse -outdir output/
```
Example: we use [pbmc8k.h5ad](https://figshare.com/s/b5737bec1ab6e1502b5a), a single cell RNA-seq dataset, as purified samples for data augmentation and put it under the ```example``` directory. So we use ```S``` for platform parameter. The calibration data is an RNA-seq expression profile ```caliexp.txt```. And we use coarse network architecture. 

#### simulation modules:
  
We have two training set simulation modules. One is DAISM_simulation which using DAISM strategy in generating mixtures. 
```
daism DAISM_simulation -platform S -caliexp ../example/caliexp.txt -califra ../example/califra.txt -aug ../example/pbmc8k.h5ad -N 16000 -testexp ../example/testexp.txt -outdir ./
```
The other is Generic_simulation which generates training set only using purified cells.

```
daism Generic_simulation -platform S -aug ../example/pbmc8k.h5ad -N 16000 -testexp ../example/testexp.txt -outdir ./
```
#### training modules:
```
daism training -trainexp ./output/DAISM_mixsam.txt -trainfra ./output/DAISM_mixfra.txt -net coarse -outdir ./
```
We use the DAISM-generated mixtures ```DAISM_mixsam.txt``` and corresponding artificial cell fractions ```DAISM_mixfra.txt``` to train the neural networks.
#### prediction modules:
```
daism prediction -testexp ../example/testexp.txt -model ./output/DAISM_model.pkl -celltype ./output/DAISM_celltypes.txt -feature ./output/DAISM_feature.txt -net coarse -outdir ./
```
