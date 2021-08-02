#!/usr/bin/env python

from setuptools import setup, find_packages

version = "1.0.5"


with open("LICENSE", encoding="UTF-8") as f:
    license = f.read()

setup(
    name="daism",
    version=version,
    description="Highly accurate cell type proportion estima-tion within silicodata augmentation and deep neural net-works",
    keywords=[
        "bioinformatics",
        "data augmentation",
        "in silico mixing",
        "deep learning",
        "single cell sequencing",
        "deconvolution",
    ],
    author="zoelin",
    author_email="linyating@stu.xmu.edu.cn",
    url="https://github.com/xmuyulab/DAISM-XMBD",
    license="MIT License",
    entry_points={"console_scripts": ["daism=daism.daism:main"]},
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.7.0",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "torch>=1.5.1",
        "anndata",
        "scanpy",
        "argh",
        "tqdm",
    ],
)