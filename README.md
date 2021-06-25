![UniTrack Logo](docs/logo.png)

--------------------------------------------------------------------------------


Paper: Do different tracking tasks require different appearance model?

[[ArXiv](https://arxiv.org)] (comming soon) [[Project Page](https://arxiv.org)] (comming soon)

UniTrack is a simple and **Uni**fied framework for versatile visual **Track**ing tasks. 

As an important problem in computer vision, tracking has been fragmented into a multitude of different experimental setups. As a consequence, the literature has fragmented too, and now the novel approaches proposed by the community are usually specialized to fit only one specific setup. To understand to what extend this specialization is actually necessary, we present UniTrack, a solution to address multiple different tracking tasks within the same framework. All tasks share the same universal [appearance model](#appearance-model). UniTrack enjoys the following advantages,

- Do **NOT** need training on a specific tracking task.

- [Good performance in existing tracking tasks]((docs/results.md)), thus can serve as strong baselines for each task.

- [Could be easily adapted to novel tasks with different setup](docs/custom_task.md).

- [Could serve as an evaluation platform to test pre-trained representations on tracking tasks](docs/study.md) (e.g. via self-supervised models).

## Tasks & Framework
![tasksframework](docs/tasksframework.png)

### Tasks
We classify existing tracking tasks along four axes: (1) Single or multiple targets; (2) Users specify targets or automatic detectors specify targets; (3) Observation formats (bounding box/mask/pose); (2) Class-agnostic or class-specific (i.e. human/vehicles). We mainly expriment on 5 tasks: **SOT, VOS, MOT, MOTS, and PoseTrack**. Task setups are summarized in the above figure.

### Appearance model
An appearance model is the only learnable component in UniTrack. It should provide universal visual representation, and is usually pre-trained on large-scale dataset in supervised or unsupervised manners. Typical examples include ImageNet pre-trained ResNets (supervised), and recent self-supervised models such as MoCo and SimCLR (unsupervised).

### Propagation and Association
Two fundamental algorithm building blocks in UniTrack. Both employ features extracted by the appearance model as input.  For propagation we adopt exiting methods such as [cross correlation](https://www.robots.ox.ac.uk/~luca/siamese-fc.html), [DCF](https://openaccess.thecvf.com/content_cvpr_2017/html/Valmadre_End-To-End_Representation_Learning_CVPR_2017_paper.html), and [mask propation](https://github.com/ajabri/videowalk). For association we employ a [simple algorithm](https://github.com/Zhongdao/Towards-Realtime-MOT) and develop a novel similarity metric to make full use of the appearance model.

## Results
Below we show results of UniTrack with a simple **ImageNet Pre-trained ResNet-18** as the appearance model. More results (other tasks/datasets, more visualization) can be found in [results.md](results.md).

### Qualitative results

**Single Object Tracking (SOT) on OTB-2015**

<img src="docs/sot1.gif" width="320"/>  <img src="docs/sot2.gif" width="320"/>

**Video Object Segmentation (VOS) on DAVIS-2017 *val* split**

<img src="docs/vos1.gif" width="320"/>  <img src="docs/vos2.gif" width="320"/>

**Multiple Object Tracking (MOT) on MOT-16 [*test* set *private detector* track](https://motchallenge.net/method/MOT=3856&chl=5)** (Detections from FairMOT)

<img src="docs/MOT1.gif" width="320"/>  <img src="docs/MOT2.gif" width="320"/>

**Multiple Object Tracking and Segmentation (MOTS) on MOTS challenge [*test* set](https://motchallenge.net/method/MOTS=109&chl=17)** (Detections from COSTA_st)

<img src="docs/MOTS1.gif" width="320"/>  <img src="docs/MOTS2.gif" width="320"/>

**Pose Tracking on PoseTrack-2018 *val* split** (Detections from LightTrack)

<img src="docs/posetrack1.gif" width="320"/>  <img src="docs/posetrack2.gif" width="320"/>

### Quantitative results

**Single Object Tracking (SOT) on OTB-2015**

| Method | SiamFC | SiamRPN | SiamRPN++ | UDT* | UDT+* | LUDT* | LUDT+* | UniTrack_XCorr* | UniTrack_DCF* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AUC | 58.2 | 63.7 | 69.6 | 59.4 | 63.2 | 60.2 | 63.9 | 55.5 | 61.8|

 \* indicates non-supervised methods

**Video Object Segmentation (VOS) on DAVIS-2017 *val* split**

| Method | SiamMask | FeelVOS | STM | Colorization* | TimeCycle* | UVC* | CRW* | VFS* | UniTrack* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| J-mean | 54.3 | 63.7 | 79.2 | 34.6 | 40.1 | 56.7 | 64.8 | 66.5 | 58.4|

 \* indicates non-supervised methods 

**Multiple Object Tracking (MOT) on MOT-16 [*test* set *private detector* track](https://motchallenge.net/method/MOT=3856&chl=5)**

| Method | POI | DeepSORT-2 | JDE | CTrack | TubeTK | TraDes | CSTrack | FairMOT* | UniTrack* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IDF-1 | 65.1 | 62.2 | 55.8 | 57.2 | 62.2 | 64.7 | 71.8 | 72.8 | 71.8|
| IDs | 805 | 781 | 1544 | 1897 | 1236 | 1144 | 1071 | 1074 | 683 |
| MOTA | 66.1 | 61.4 | 64.4 | 67.6 | 66.9 | 70.1 | 70.7 | 74.9 | 74.7|

 \* indicates methods using the same detections

**Multiple Object Tracking and Segmentation (MOTS) on MOTS challenge [*test* set](https://motchallenge.net/method/MOTS=109&chl=17)**

| Method | TrackRCNN | SORTS | PointTrack | GMPHD | COSTA_st* | UniTrack* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| IDF-1 | 42.7 | 57.3 | 42.9 | 65.6 | 70.3 | 67.2 |
| IDs | 567 | 577 | 868 | 566 | 421 | 622 | 
| sMOTA | 40.6 | 55.0 | 62.3 | 69.0 | 70.2 | 68.9 | 

 \* indicates methods using the same detections

**Pose Tracking on PoseTrack-2018 *val* split**

| Method | MDPN | OpenSVAI | Miracle | KeyTrack | LightTrack* | UniTrack* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| IDF-1 | - | - | - | - | 52.2 | 73.2 |
| IDs | - | - | - | - | 3024 | 6760 | 
| sMOTA | 50.6 | 62.4 | 64.0 | 66.6 | 64.8 | 63.5 | 

 \* indicates methods using the same detections


## Getting started


## Demo


## Update log
[2021.6.24]: Start writing docs, please stay tuned!

## Acknowledgement
[VideoWalk](https://github.com/ajabri/videowalk) by Allan A. Jabri

[SOT code](https://github.com/JudasDie/SOTS) by Zhipeng Zhang