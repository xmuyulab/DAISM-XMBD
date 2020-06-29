## DAISM-DNN

We propose data augmentation throughin silicomixing with deep neural networks (DAISM-DNN) to achieve highly accurate andunbiased  immune-cell  proportion  estimation  from  bulk  RNA  sequencing  (RNA-seq)  data. Our method tackles the batch effect problem by creating a data-specific training dataset froma small subset of samples with ground truth cell proportions which is further augmentedwith  publicly  available  RNA-seq  data  from  purified  cells  or  single-cell  RNA-seq  (scRNA-seq) data.

A pre-print describing the method is available at Biorxiv:
 [DAISM-DNN: Highly accurate cell type proportion estima-tion within silicodata augmentation and deep neural net-works](https://www.biorxiv.org/content/10.1101/2020.03.26.009308v2)
 
 ### Dependencies
 It is recommended to create a new conda environment:
 ```
 conda create -n DAISM python=3.7

# Activate this environment:
conda activate DAISM
 ```
 ```
argh (0.26.2) 
anndata (0.6.22)
scanpy (1.4.3)
sklearn (0.0)
scikit-learn (0.21.2)
scipy (1.3.0)
python (3.7.3)
numpy (1.16.3)
pytorch (1.0.1)
pandas (0.25.1)
```
We provide a docker image with DAISM-DNN installed:
[DAISM-DNN]()

### Cell Types Supported

|Granularity|Cell types|
|---|---|
|Coarse-garined|B.cells|
||CD4.T.cells|
||CD8.T.cells|
||Monocytes|
||NK.cells|
||Neutrophils|
|Fine-grained|Naive.B.cells|
||Memory.B.cells|
||Naive.CD4.T.cells|
||Memory.CD4.T.cells|
||Naive.CD8.T.cells|
||Memory.CD8.T.cells|
