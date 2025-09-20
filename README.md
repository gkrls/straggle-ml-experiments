### Torch vision models

[https://docs.pytorch.org/vision/main/models.html](https://docs.pytorch.org/vision/main/models.html)
[https://catalog.ngc.nvidia.com/orgs/nvidia/resources/efficientnet_for_pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/efficientnet_for_pytorch)

### Model statistics

#### Gradient size:

- Resnet50 ~98MB
- Resnet101 ~178MB
- Resnet152 ~240MB

#### Rough epoch times on HPDC cluster

- Resnet50, deterministic, gloo, batch_size=128 -- 1025sec ~208 images/sec
- Resnet50, deterministic, nccl, batch_size=128 -- 1000sec ~215 images/sec
- Resnet101, deterministic, nccl, batch_size=128 -- 1500sec
- Resnet154, deterministic, nccl, batch_size=128 -- 2160sec
- VGG11, deterministic, gloo, batch_size=128 -- 1600sec ~133 images/sec
- Densenet121, deterministic, gloo, batch_size=128 -- 1021sec ~208 images/sec
- EfficientNet_b0, deterministic, gloo, batch_size=128 -- 1550sec ~137.2 img/s



```bash
./hpdc-launch.sh --keep openwebtext -f hosts.txt -S nfs-rsync.sh /nfs0/datasets/openwebtext /home/gks/datasets
./hpdc-launch.sh --keep openwebtext -f hosts.txt -- tar -xvf /home/gks/datasets/openwebtext/openwebtext.tar.xz -C /home/gks/datasets/openwebtext 
```

#### Setup pyenv:

Pyenv can be setup for easier python version control. It can be done by running:

```bash
./setup_pyenv.sh
source "$HOME/.bashrc"
```