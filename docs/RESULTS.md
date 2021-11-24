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
