<div align="center">

# BBCPy_DNN

[![License](https://img.shields.io/github/license/ashleve/lightning-hydra-template?color=blue)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#how-to-run">How to Run</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## About The Project

BBCPy_DNN is an advanced component of the BBCPy toolbox, designed to facilitate the development and exploration of Deep Neural Networks (DNNs) within the PyTorch framework. Leveraging PyTorch's dynamic computation graph, PyTorch Lightning for high-level framework functionalities, and Hydra for dynamic management of configuration files, BBCPy_DNN offers researchers and developers a flexible and efficient interface to develop and reproduce DNNs, particularly for BCI applications.

### Features

- **Intuitive Interface:** Simplifies the complexity of DNN development, making it more accessible, especially for newcomers.
- **Streamlined Training Process:** Utilizes PyTorch Lightning to balance user-friendliness with high-level functionality for deep learning research.
- **Supports ML Lifecycle:** From data extraction and exploration to model deployment, supporting continuous improvement and superior model performance.
- **Flexible Configuration:** Integrated with Hydra for dynamic management of configuration files, allowing easy optimization and workflow management without altering source code.

### Built With

BBCPy_DNN is built with the help of the following frameworks: 
* [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
* [Optuna](https://optuna.org/)
* [MLflow](https://mlflow.org/)
* [Hydra](https://hydra.cc/)

### Prerequisites

- **BBCPy toolbox:** A Python-based toolbox for Brain-Computer Interface (BCI) research.
- **Motor Imagery Dataset:** Supports various datasets, including the Continuous SMR BCI dataset, which includes EEG data from 62 healthy individuals for BCI studies.

#### To download the Continuous SMR BCI dataset (351GB):

```bash
mkdir -p ~/data 
wget https://figshare.com/ndownloader/articles/13123148/versions/1
unzip 1
```


## Installation

#### Pip

```bash
# clone project
git clone https://github.com/mralioo/BBCPy_DNN.git
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
git clone https://github.com/mralioo/BBCPy_DNN.git
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


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
## Contact

* **Ali Alouane** (ali.alouane@campus.tu-berlin.de)


## Acknowledgements

We want to give many thanks to our project supervisor **Dr. Daniel Miklody**, **Dr. Oleksandr Zlatov**, and the Neurotechnology group at TU Berlin.