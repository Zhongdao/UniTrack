# MODEL ZOO


### Prepare apperance models
One beneficial usage of UniTrack is that it allows easy evaluation of pre-trained models (as appearance models) on diverse tracking tasks. By far we have tested the following models, mostly self-supervised pre-trained:

| Pre-training Method | Architecture |Link | 
| :---: | :---: | :---: |
| ImageNet classification | ResNet-50 | torchvision |
| InsDist| ResNet-50 | [Google Drive](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth)|
| MoCo-V1| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar)|
| PCL-V1| ResNet-50 |[Google Drive](https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar)|
| PIRL| ResNet-50 | [Google Drive](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth)|
| PCL-V2| ResNet-50 | [Google Drive](https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar)|
| SimCLR-V1| ResNet-50 |[Google Drive](https://drive.google.com/file/d/1RdB2KaaXOtU2_t-Uk_HQbxMZgSGUcy6c/view?usp=sharing)|
| MoCo-V2| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar)|
| SimCLR-V2| ResNet-50 |[Google Drive](https://drive.google.com/file/d/1NSCrZ7MaejJaOS7yA3URtbubxLR-fz5X/view?usp=sharing)|
| SeLa-V2| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar)|
| InfoMin| ResNet-50 | [Google Drive](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth)|
| BarlowTwins| ResNet-50 | [Google Drive](https://drive.google.com/file/d/1iXfAiAZP3Lrc-Hk4QHUzO-mk4M4fElQw/view?usp=sharing)|
| BYOL| ResNet-50 | [Google Drive](https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl)|
| DeepCluster-V2| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar)|
| SwAV| ResNet-50 |[Google Drive](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar)|
| PixPro| ResNet-50 |[Google Drive](https://drive.google.com/file/d/1u172sUx-kldPvrZzZxijciBHLMiSJp46/view?usp=sharing)|
| DetCo| ResNet-50 | [Google Drive](https://drive.google.com/file/d/1ahyX8HEbLUZXS-9Jr2GIMWDEZdqWe1GV/view?usp=sharing)|
| TimeCycle| ResNet-50 |[Google Drive](https://drive.google.com/file/d/1WUYLkfowJ853RG_9OhbrKpb3r-cc-cOA/view?usp=sharing)|
| ImageNet classification | ResNet-18 |torchvision|
| Colorization + memory| ResNet-18 | [Google Drive](https://drive.google.com/file/d/1gWPRgYH70t-9uwj0EId826ZxFdosbzQv/view?usp=sharing)|
| UVC| ResNet-18 |[Google Drive](https://drive.google.com/file/d/1nl0ehS8mvE5PUBOPLQSCWtrmFmS0-dPX/view?usp=sharing)|
| CRW| ResNet-18 |[Google Drive](https://drive.google.com/file/d/1C1ujnpFRijJqVD3PV7qzyYwGSWoS9fLb/view?usp=sharing)|

After downloading an appearance model, please place it under `$UNITRACK_ROOT/weights`. A large part of the model checkpoints are adopted from [ssl-transfer](https://github.com/linusericsson/ssl-transfer), many thanks to [linusericsson](https://github.com/linusericsson)!

### Test your own pre-trained models as appearance models
If your model uses the standard ResNet architecture, you can directly test it using UniTrack without additional modifications. If you use ResNet but the parameter names are not consistent with the standard naming, you can simply rename parameter groups and load your weights into the standard ResNet. If you are using other architectures, it is also possible to test it with UniTrack. You may need a little hack: just remember to let the model output 8x down-sampled feature maps. You can check out `models/hrnet.py` for an example. 
