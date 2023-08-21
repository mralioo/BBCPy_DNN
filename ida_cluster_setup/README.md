
# Apptainer

## Installation

### Install system dependencies

https://github.com/apptainer/apptainer/blob/main/INSTALL.md

```bash
# Ensure repositories are up-to-date
sudo apt-get update
# Install debian packages for dependencies
sudo apt-get install -y \
    build-essential \
    libseccomp-dev \
    pkg-config \
    uidmap \
    squashfs-tools \
    squashfuse \
    fuse2fs \
    fuse-overlayfs \
    fakeroot \
    cryptsetup \
    curl wget git
```

### Install Go

```bash
# Download Go
wget https://dl.google.com/go/go1.14.2.linux-amd64.tar.gz
# Extract Go
sudo tar -C /usr/local -xzf go1.14.2.linux-amd64.tar.gz
# Add Go to PATH
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
# Reload bashrc
source ~/.bashrc
```

### Install Singularity

```bash
# Clone the repo
git clone https://github.com/apptainer/apptainer.git
cd apptainer

# Build Singularity

./mconfig
cd ./builddir
make
sudo make install

apptainer --version
```


## Usage
### Define an Apptainer

```bash
# Create a directory for the Apptainer
Bootstrap: docker
From: python:3.10

%files
    $PWD/requirements.txt requirements.txt

%post
    pip install --root-user-action=ignore -r requirements.txt
```

### Build an Apptainer

```bash
# hydra task to build an apptainer
srun --partition=cpu-2h --pty bash

# build the apptainer
apptainer build python_container.sif python_container.def
```
### Run an Apptainer

```bash
apptainer run --nv python_container.sif python -c "import torch; print(torch.cuda.is_available())"
```

### Update an Apptainer
use previous container as a starting point by bootstrapping from it.

```bash
Bootstrap: localimage
From: /opt/apps/pytorch-2.0.1-gpu.sif

%post
    pip install --root-user-action=ignore scikit-learn
```

#SquashFS
