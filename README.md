## DAISM-DNN

We propose data augmentation throughin silicomixing with deep neural networks (DAISM-DNN) to achieve highly accurate andunbiased  immune-cell  proportion  estimation  from  bulk  RNA  sequencing  (RNA-seq)  data. Our method tackles the batch effect problem by creating a data-specific training dataset froma small subset of samples with ground truth cell proportions which is further augmentedwith  publicly  available  RNA-seq  data  from  purified  cells  or  single-cell  RNA-seq  (scRNA-seq) data.

A pre-print describing the method is available at Biorxiv:
 [DAISM-DNN: Highly accurate cell type proportion estima-tion within silicodata augmentation and deep neural net-works](https://www.biorxiv.org/content/10.1101/2020.03.26.009308v2)
 
 ### Dependencies
 It is recommended to create a new conda environment:
 ```
 conda create -n DAISM_DNN python=3.7

# Activate this environment:
conda activate DAISM_DNN
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

### Usage
First of all, we should:
change directory (cd) to daism_dnn folder and call daismIndex module help for details
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

-platform    string    The platform of [calibration data] + [purified data for agumentation], [Rs]: RNA-seq TPM + scRNA, [Rt]: RNA-seq TPM + TPM, [Ms]: Microarray + scRNA
                        
-caliExp     string   The calibration samples expression file

-caliFra     string   The calibration samples ground truth file

-pureExp     string   The purified samples expression (h5ad)

-simNum      int      The number of simulation samples

-inputExp    string   The test samples expression file

-outputDir   string   The directory of result files
```

- simulation modules:
```

python daism_dnn.py simulation -h

python daism_dnn.py simulation -platform Rs -caliExp path1 -caliFra path2 -pureExp path3 -simNum 16000 -outputDir dir1

Required arguments:

-platform string The platform of [calibration data] + [purified data for agumentation], [Rs]: RNA-seq TPM + scRNA, [Rt]: RNA-seq TPM + TPM, [Ms]: Microarray + scRNA

-caliExp     string   The calibration samples expression file

-caliFra     string   The calibration samples ground truth file

-pureExp     string   The purified samples expression (h5ad)

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
