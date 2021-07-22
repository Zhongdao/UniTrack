# Run evaluation on multiple tasks

### Prepare config file

To evaluate an apperance model on multiple tasks, first you need to prepare a config file `${EXP_NAME}.yaml` and place it under the `config/` folder. We provide several example config files: 
   1. `crw_resnet18_s3.yaml` : Self-supervised model trained with Contrastive Random Walk [1], ResNet-18 stage-3 features.
   2. `imagenet_resnet18_s3.yaml`: ImageNet pre-trained model, ResNet-18 stage-3 features.
   3. `crw_resnet18_s3_womotion.yaml` : Model same as 1 but motion cues are discarded in association type tasks. This way, distinctions between different representations are better highlighted and  potential confounding factors are avoided.
   4. `imagenet_resnet18_s3_womotion.yaml`: Model same as 2, motion cues are discared in association type tasks.
   

### Note for the config file

When you are testing a new model, please take care to make sure the following fields in the config file are correct:

```yaml
common:
    # Experiment name, an identifier.
    exp_name: crw_resnet18_s3   
    
    # Model type, currently support:
    # ['imagenet18', 'imagenet50', 'imagenet101', 'random18', 'random50',
    # 'imagenet_resnext50', 'imagenet_resnext101'
    # 'byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1', 'moco-v2',
    # 'pcl-v1', 'pcl-v2','pirl', 'sela-v2', 'swav', 'simclr-v1', 'simclr-v2',
    # 'pixpro', 'detco', 'barlowtwins', 'crw', 'uvc', 'timecycle']
    model_type: crw                    

    # For ResNet architecture, remove layer4 means output layer3 features
    remove_layers: ['layer4']             
    
    # Be careful about this
    im_mean: [0.4914, 0.4822, 0.4465]        
    im_std: [0.2023, 0.1994, 0.2010]
    
    # Path to the model weights.
    resume: 'weights/crw.pth'
    
mot:
    # The single-frame observations. should correspond to a folder ${mot_root}/obs/${obid}
    obid: 'FairMOT'
    # Dataset root
    mot_root: '/home/wangzd/datasets/MOT/MOT16'
    # There is no validation set, so by default we test on the train split. 
    
mots:
    # The single-frame observations. should correspond to a folder ${mots_root}/obs/${obid}
    obid: 'COSTA'
    # Dataset root
    mots_root: '/home/wangzd/datasets/GOT/MOTS'
    # There is no validation set, so by default we test on the train split.  
    
posetrack:
    # The single-frame observations. should correspond to a folder ${mots_root}/obs/val/${obid}
    obid: 'lighttrack_MSRA152
    # Dataset root
    data_root: '/home/wangzd/datasets/GOT/Posetrack2018'
    # There is a validation set, by default we test on the val split.
    split: 'val'
                                 
```

For other arguments, just refer to `crw_resnet18_s3.yaml` or `crw_resnet18_s3_womotion.yaml`.


### Run

Suppose the current path is `$UNITRACK_ROOT`, you can run multiple tasks with a single command:

```shell
./eval.sh $EXP_NAME $GPU_ID
```

You will obtain a set of summaries of quantitative results under `results/summary`, and also visualizations of all results under `results`


[1]. Jabri, Allan, Andrew Owens, and Alexei A. Efros. "Space-time correspondence as a contrastive random walk." In NeurIPS, 2020.