## DAISM-DNN

We propose data augmentation through in silico mixing with deep neural networks (DAISM-DNN) to achieve highly accurate and unbiased immune-cell proportion estimation from bulk  RNA sequencing (RNA-seq) data. Our method tackles the batch effect problem by creating a data-specific training dataset from a small subset of samples with ground truth cell proportions which is further augmented with publicly available RNA-seq data from purified cells or single-cell RNA-seq (scRNA-seq) data.

A pre-print describing the method is available at bioRxiv:
 [DAISM-DNN: Highly accurate cell type proportion estima-tion within silicodata augmentation and deep neural net-works](https://www.biorxiv.org/content/10.1101/2020.03.26.009308v2)
 
### Dependencies
It is recommended to create a new conda environment:
```
conda create -n DAISM_DNN python=3.7

# Activate this environment:
conda activate DAISM_DNN
```
Install the dependencies below:
```
python (v3.7.7)
pytorch (v1.5.1)
pandas (v1.0.5)
numpy (v1.18.1)
scikit-learn (v0.23.1)
argh (v0.26.2) 
anndata (v0.7.3)
scanpy (v1.5.1)
```
We provide a docker image with DAISM-DNN installed:
[https://hub.docker.com/r/zoelin1130/daism_dnn](https://hub.docker.com/r/zoelin1130/daism_dnn)

Pull the docker image:
```
docker pull zoelin1130/daism_dnn:1.0
```
Create a container (GPU):
```
docker run --gpus all -i -t run_daism -v example:/workspace/example/ zoelin1130/daism_dnn:1.0 /bin/bash
```
Create a container (CPU):
```
docker run -i -t run_daism -v example:/workspace/example/ zoelin1130/daism_dnn:1.0 /bin/bash
```
```run_daism```is your container name. ```example```means the directory of your data.

### Cell Types Supported
The example we provide contains the following cell types. The purified dataset for data augmentation can be downloaded from:[https://figshare.com/s/b5737bec1ab6e1502b5a](https://figshare.com/s/b5737bec1ab6e1502b5a)

pbmc8k.h5ad contains 5 cell types: B.cells, CD4.T.cells, CD8.T.cells, monocytic.lineage, NK.cells.

pbmc8k_fine.h5ad contains 11 cell types: naive.B.cells, memory.B.cells, naive.CD4.T.cells, memory.CD4.T.cells,naive.CD8.T.cells, memory.CD8.T.cells, regulatory.T.cells, monocytes, macrophages, myeloid.dendritic.cells, NK.cells.

RNA-seq_TPM.h5ad contains 27 cell types: B.cells, monocytic.lineage, dendritic.cells, naive.CD4.T.cells, NK.cells, macrophage, regulatory.T.cells, naive.B.cells, memory.B.cells, neutrophils, CD4.T.cells, Central.memory.CD4.T.cells, Effector.memory.CD4.T.cells, endothelial.cells, fibroblasts, memory.CD4.T.cells, basophils, myeloid.dendritic.cells, CD8.T.cells, Central.memory.CD8.T.cells, naive.CD8.T.cells, Effector.memory.CD8.T.cells, eosinophils, follicular.helper.T.cells, gamma.delta.T.cells, plasma.cells,memory.CD8.T.cells.

Note: each cell type needs to be named according to above format.

DAISM-DNN can support the prediction of any cell types, as long as calibration samples with ground truth and purified expression profiles of corresponding cell types are provided.

### Usage
In our example below, we set working directory to daism_dnn. Use -h to print out help information on DAISM-DNN modules.
```
cd daism_dnn
python daism_dnn.py -h
```

DAISM-DNN consists of four modules:

- DAISM-DNN modules: 
```
python daism_dnn.py DAISM-DNN -h

python daism_dnn.py DAISM-DNN -platform Rs -caliExp path1 -caliFra path2 -pureExp path3 -simNum 16000 -outputDir dir1 -inputExp path4

Required arguments:

-platform    string   The platform of [calibration data] + [purified data for augmentation], [Rs]: RNA-seq TPM + scRNA, [Rt]: RNA-seq TPM + TPM, [Ms]: Microarray + scRNA
                        
-caliExp     string   The calibration samples expression file

-caliFra     string   The calibration samples ground truth file

-pureExp     string   The purified samples expression (h5ad)

-simNum      int      The number of simulation samples

-inputExp    string   The test samples expression file

-outputDir   string   The directory of result files
```

Example: we use [pbmc8k.h5ad](https://figshare.com/s/b5737bec1ab6e1502b5a), a single cell RNA-seq dataset, as purified samples for data augmentation. Put it under the ```example``` directory. The calibration data is an RNA-seq expression profile ```example_caliexp.txt```. So we use ```Rs``` for platform parameter.

```
python daism_dnn.py DAISM-DNN -platform Rs -caliExp ../example/example_caliexp.txt -caliFra ../example/example_califra.txt -pureExp ../example/pbmc8k.h5ad -simNum 16000 -outputDir ../output/ -inputExp ../example/example_testexp.txt
```
If no calibration samples are available, the training data simulation mode should be changed from ```DAISM``` to ```puremix```.

```
python daism_dnn.py DAISM-DNN -platform Rs -pureExp ../example/pbmc8k.h5ad -simNum 16000 -outputDir ../output/ -inputExp ../example/example_testexp.txt
```


- simulation modules:
```
python daism_dnn.py simulation -h

python daism_dnn.py simulation -platform Rs -caliExp path1 -caliFra path2 -pureExp path3 -inputExp path4 -simNum 16000 -outputDir dir1

Required arguments:

-platform    string   The platform of [calibration data] + [purified data for augmentation], [Rs]: RNA-seq TPM + scRNA, [Rt]: RNA-seq TPM + TPM, [Ms]: Microarray + scRNA

-caliExp     string   The calibration samples expression file

-caliFra     string   The calibration samples ground truth file

-pureExp     string   The purified samples expression (h5ad)

-inputExp    string   The test samples expression file

-simNum      int      The number of simulation samples

-outputDir   string   The directory of simulated output files
```

- training modules:
```
python daism_dnn.py training -h 

python daism_dnn.py training -trainExp path1 -trainFra path1 -outputDir dir1

Required arguments:

-trainExp    string   The simulated samples expression file

-trainFra    string   The simulated samples ground truth file

-outputDir   string   The directory of output files
```

- prediction modules:
```
python daism_dnn.py prediction -h 

python daism_dnn.py prediction -inputExp path1 -model path2 -cellType path3 -feature path4 -outputDir dir1

Required arguments:

-inputExp   string    The test samples expression file

-model      string    The deep-learing model file trained by DAISM-DNN

-cellType   string    Model celltypes

-feature    string    Model features

-outputDir  string    The directory of output result files
```
