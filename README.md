This repository provides code and annotations for the paper "Machine Learning Based Analysis of Finnish World War II Photographers". The preprint is available [here].

## Getting Started
This repository mainly relies on the following libraries:
* python 3.5
* keras 2.1.6
* tensorflow 1.10
* opencv 3.4.2

If you use conda, all dependancies can be installed by running:
```sh
$ conda create -n "ww2-analysis" -y
$ conda activate ww2-analysis
$ ./requirements.sh
```
This repository relies on [Mask RCNN], [Yolov3], [SSD], and [RetinaNet] implementations. Please refer to each of them for installation instructions. Download the pretrained models from each repository, and put them into *./src/models/*. More specifically, [SSD model], [Mask RCNN model], [RetinaNet model]. For Yolo model, download the weights from the github repository and convert them to keras format according to instructions.
In the end, your project repository should like this:
```sh
${project_dir}/
├── annotations
├── classification
│   └── models
│       └── model.h5
├── detection
├── src
│   ├── keras_yolo3
│   ├── mask_rcnn
│   ├── retinanet
│   ├── ssd
│   └── models
│       ├── yolo.h5
│       ├── resnet50_coco_best_v2.0.3.h5
│       ├── VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.h5
│       └── mask_rcnn_coco.h5
```
## Detection
If you want to apply the detection pipeline and analysis to your own data, specify your read and write directories: 
```sh
$ python run_detections.py -r=[PATH_TO_IMAGES] -s=[SAVE_DIR] -i
$ python run_mask.py -r=[PATH_TO_IMAGES] -s=[SAVE_DIR] -i
$ python aggregate.py -r=[PATH_TO_IMAGES] -s=[SAVE_DIR] -i
```
This way *run_detections.py* runs SSD, Yolov3, and RetinaNet detectors, *run_mask.py* performs detection with Mask-RCNN, and *aggregate.py* agregates the produced bounding boxes and perform further analysis (distance estimation, etc.). The *-i* argument specifies wether the processed images with bounding boxes on them should be saved.
The result with key information will be stored in *result.txt* under your [SAVE_DIR].  More detailed results are stored in *photographer_results_dict.json*.
## Classification

For photographer classification, the cross-validation splits used in the paper can be found from *./classification/partition.pkl*.
To train your own model:
```sh
$ python train.py -r=[PATH_TO_IMAGES]
```
The parameters such as learning rate, batch size, number of epochs, etc., can be specified in train.py.
For inference, first download the [pretrained model] and put it to *./classification/models/* folder. Then run:
```sh
$ python test.py -r=[PATH_TO_IMAGES]
```
This will create the file *./classification/result.txt* with obtained accuracy and the confusion matrix.
## Annotations
The bounding box annotations for the photos are available from the file *bounding_boxes.txt*, where each row corresponds to a separate detection. The annotations follow the following format:
```
fileid top left bottom right confidence_score label
fileid2 top2 left2 bottom2 right2 confidence_score2 label2
...
```
The images are available from [here.] 
### 
If you find our work useful, please kindly cite our paper as:
```
@article{chumachenko2019machine,
  title={Machine Learning Based Analysis of Finnish World War II Photographers},
  author={Chumachenko, Kateryna and M{\"a}nnist{\"o}, Anssi and Iosifidis, Alexandros and Raitoharju, Jenni},
  journal={arXiv preprint arXiv:1904.09811},
  year={2019}
}
```
## References
* Mask RCNN: https://github.com/matterport/Mask_RCNN
* Yolov3: https://github.com/qqwweee/keras-yolo3
* RetinaNet: https://github.com/fizyr/keras-retinanet
* SSD: https://github.com/pierluigiferrari/ssd_keras

[here]: <https://arxiv.org/abs/1904.09811>
[Mask RCNN]: <https://github.com/matterport/Mask_RCNN>
[Yolov3]: <https://github.com/qqwweee/keras-yolo3>
[SSD]: <https://github.com/pierluigiferrari/ssd_keras>
[RetinaNet]: <https://github.com/fizyr/keras-retinanet>
[pretrained model]: <https://drive.google.com/open?id=19-m0vSHMQHLzH2UQtEkJ-gKXUgevas_R>
[SSD model]: <https://drive.google.com/file/d/1a-64b6y6xsQr5puUsHX_wxI1orQDercM/view>
[Mask RCNN model]: <https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5>
[RetinaNet model]: <https://github.com/fizyr/keras-retinanet/releases/download/0.2/resnet50_coco_best_v2.0.3.h5>
[here.]: <http://sa-kuva.fi/>

