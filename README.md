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
 Install the dependencies below:
 ```
python (v3.7.7)
pytorch (v1.5.1)
pandas (v1.0.5)
numpy (v1.18.1)
sklearn (v0.0)
scikit-learn (v0.23.1)
argh (v0.26.2) 
anndata (v0.7.3)
scanpy (v1.5.1)
scipy (v1.5.0)
```
We provide a docker image with DAISM-DNN installed:
[https://hub.docker.com/r/zoelin1130/daism_dnn](https://hub.docker.com/r/zoelin1130/daism_dnn)

Pull the docker image:
```
docker pull zoelin1130/daism_dnn:1.0
```
Create a container (GPU):
```
docker run --gpus all -i -t run_daism zoelin1130/daism_dnn:1.0 /bin/bash
```
Create a container (CPU):
```
docker run -i -t run_daism zoelin1130/daism_dnn:1.0 /bin/bash
```
```run_daism```is your container name.
Add data to container:
```
docker cp exampledata run_daism:/workspace/
```
```exampledata```means the directory or files of your data.
### Cell Types Supported

|Granularity|Cell types|
|---|---|
|Coarse-garined|B.cells|
||CD4.T.cells|
||CD8.T.cells|
||Monocytes|
||NK.cells|
||Neutrophils|


### Usage
First of all, we should:
change directory (cd) to daism_dnn folder and call daismIndex module help for details
```
cd daism_dnn
python daism_dnn.py -h
```
If you use docker to run our software:

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
