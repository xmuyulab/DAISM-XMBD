#!/bin/bash

apt-get update
apt-get install less
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda
conda install argh
conda install anndata
conda install scikit-learn
#pip install sklearn
pip install scanpy
