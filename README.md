<div align="center">

# BBCPy_DNN

[![License](https://img.shields.io/github/license/ashleve/lightning-hydra-template?color=blue)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description
BBCPy_DNN is an advanced component of the BBCPy toolbox, designed to facilitate the development and exploration of Deep Neural Networks (DNNs) within the PyTorch framework. By leveraging PyTorch's dynamic computation graph and PyTorch Lightning, BBCPy_DNN offers an intuitive and efficient interface for model training, making it suitable for rapid prototyping and scalable for large-scale applications.


## Features

- **Intuitive Interface:** Simplifies the complexity of DNN development, making it more accessible, especially for newcomers.
- **Streamlined Training Process:** Utilizes PyTorch Lightning to balance user-friendliness with high-level functionality for deep learning research.
- **Supports ML Lifecycle:** From data extraction and exploration to model deployment, supporting continuous improvement and superior model performance.
- **Flexible Configuration:** Integrated with Hydra for dynamic management of configuration files, allowing easy optimization and workflow management without altering source code.


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
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
