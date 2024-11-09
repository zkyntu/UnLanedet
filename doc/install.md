## Installation step
<font size=4> In this documentation, we introduce the installation step by step.

### Docker installation (recommendation)
<font size=4>1. Pull the Pytorch docker image.

<font size=3>

```Shell
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

<font size=4>2. Create the container

<font size=3>

```Shell
docker create --gpus all --name ldet -v $PWD:/home --shm-size 20G --network=host -it pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel /bin/bash

docker start ldet

docker exec -it ldet bash
```

<font size=4>3. Install the requirements

<font size=3>

```Shell
# git clone https://github.com/zkyntu/UnLanedet.git
cd /home/UnLanedet
pip install -r requirements.txt
pip install numpy==1.23.1
pip install hydra-core --upgrade
python setup.py build develop
```

### Conda Installation
<font size=3>

```Shell
# git clone https://github.com/zkyntu/UnLanedet.git
conda create -n unlanedet python=3.9 -y
conda activate unlanedet
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
cd UnLanedet
pip install -r requirements.txt
pip install numpy==1.23.1
pip install hydra-core --upgrade
python setup.py build develop
```
