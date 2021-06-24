![UniTrack Logo](docs/logo.png)

--------------------------------------------------------------------------------


Paper: Do different tracking tasks require different appearance model?

[[ArXiv]((https://arxiv.org))] (comming soon) [[Project Page]((https://arxiv.org))] (comming soon)

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

## Getting started

## Demo


## Results

Please see [results.md](docs/results.md).

## Update log
[2021.6.24]: Start writing docs, please stay tuned!

## Acknowledgement
[VideoWalk](https://github.com/ajabri/videowalk) by Allan A. Jabri

[SOT code](https://github.com/JudasDie/SOTS) by Zhipeng Zhang