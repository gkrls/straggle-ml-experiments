### Torch vision models

[https://docs.pytorch.org/vision/main/models.html](https://docs.pytorch.org/vision/main/models.html)



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