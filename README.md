<div align="center">

# Erase to Enhance:Data-Efficient Machine Unlearning in MRI Reconstruction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/YuyangXueEd/ReconHydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Preprint](http://img.shields.io/badge/arxiv-2405.15517-B31B1B.svg)](https://arxiv.org/abs/2405.15517)
[![Conference](http://img.shields.io/badge/MIDL-2024-aa0000.svg)](https://openreview.net/forum?id=FmCscsj7Ey)

</div>

## News

- 2024/04/06: Our paper has been accpeted in MIDL 2024!🎉🎉 

## Abstract

Machine unlearning is a promising paradigm for removing unwanted data samples from a trained model, towards ensuring compliance with privacy regulations and limiting harmful biases. Although unlearning has been shown in, e.g., classification and recommendation systems, its potential in medical image-to-image translation, specifically in image recon-struction, has not been thoroughly investigated. This paper shows that machine unlearning is possible in MRI tasks and has the potential to benefit for bias removal. We set up a protocol to study how much shared knowledge exists between datasets of different organs, allowing us to effectively quantify the effect of unlearning. Our study reveals that combining training data can lead to hallucinations and reduced image quality in the reconstructed data. We use unlearning to remove hallucinations as a proxy exemplar of undesired data removal. Indeed, we show that machine unlearning is possible without full retraining. Furthermore, our observations indicate that maintaining high performance is feasible even when using only a subset of retain data. We have made our code publicly accessible.


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YuyangXueEd/ReconUnlearning
cd ReconUnlearning

# [OPTIONAL] create conda environment
conda create -n unrecon python=3.10
conda activate unrecon

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YuyangXueEd/ReconUnlearning
cd ReconUnlearning

# create conda environment and install dependencies
conda env create -f environment.yaml -n unrecon

# activate conda environment
conda activate unrecon
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
