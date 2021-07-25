# Installation

### Requirements
* Nvidia device with CUDA 
* Python 3.7+
* PyTorch 1.7.0+
* torchvision 0.8.0+
* Other python packages in requirements.txt

### Code installation

#### (Recommended) Install with conda

Install conda from [here](https://repo.anaconda.com/miniconda/), Miniconda3-latest-(OS)-(platform).
```shell
# 1. Create a conda virtual environment.
conda create -n unitrack python=3.7 -y
conda activate unitrack

# 2. Install PyTorch
conda install pytorch==1.7.0 torchvision cudatoolkit

# 3. Get UniTrack
git clone https://github.com/Zhongdao/UniTrack.git
cd UniTrack

# 4. Install ohter dependency
conda install --file requirements.txt 
pip install cython_bbox==0.1.3
python setup.py

```

