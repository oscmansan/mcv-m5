# M5 Project: Scene Understanding for Autonomous Vehicles

The goal of this project is to learn the basic concepts and techniques to build deep neural networks to detect, segment and recognize specific objects. These techniques will be applied to environment perception for autonomous vehicles, whereby the classes of interest with regard the three tasks will be pedestrians, vehicles, road, roadside, etc.

## Team 6 members

* Alba Herrera (albaherrerapalacio at gmail dot com)
* Jorge López (jorgelopezfueyo at gmail dot com)
* Oscar Mañas
* Pablo Rodríguez (pablorodriper at gmail dot com)

## Running instructions

```
pip3 install -r requirements.txt
python3 src/main.py --config_file CONFIG_FILE
```

## Directory structure

```
.
├── config              # framework configurations
├── devkit_kitti_txt
├── docs                # summaries of nn systems
│   ├── resnet.md
│   └── vgg.md
├── fonts
├── jobs                # jobs to schedule in the SLURM cluster
├── README.md
├── requirements.txt    # python dependencies
└── src
    ├── config
    ├── dataloader
    ├── main.py
    ├── metrics
    ├── models
    ├── tasks
    └── utils
```

## Weekly Deliverables

### Week 2 - Object Recognition
- [Summary](./docs/week2.md) 
- [Slides](https://docs.google.com/presentation/d/1e6U8LvV8q_5QeuToiP9Zytm0M13JhbPFJgBwRQDjbQw/edit?usp=sharing)
- Model weights:
    - [OscarNet on BelgiumTSC](https://drive.google.com/uc?export=download&id=1KiY8Lqg4y3A9inW8OYOn1Z-lndlB3yIJ)

To run the framework with the weights above, execute:
```
python3 src/main.py --config_file config/oscarnet_tt100k_pretrained.yml
```

### Weeks 3,4 - Image Semantic Segmentation
- [Summary](./docs/week3.md) 
- [Slides](https://docs.google.com/presentation/d/1Tw2_rM0kb7KlDa2SXAh9ICdI2R2cIrZHyfTIjgGYIwc/edit?usp=sharing)
- Model weights:
    - [PSPNet50 on CamVid](https://drive.google.com/uc?export=download&id=1RhUOsQdC6zoomOSCUpkY-gMRaX814M4v)
    - [PSPNet101 on CamVid](https://drive.google.com/uc?export=download&id=1NsFPgiJlvNe4JElNZtAZ8ZjVYgztT3p4)

### Weeks 5,6 - Object Detection Segmentation
- [Repository](https://github.com/oscmansan/maskrcnn-benchmark)
- [Summary](./docs/week5.md) 
- [Slides](https://docs.google.com/presentation/d/1rSKkwpA3lUG5QKRCXHqQDit7M6jYonH-cRXTqPqd-IE/edit#slide=id.g5270bddbcf_0_436)
- Model weights:
    - [RetinaNet on TT100K](https://drive.google.com/uc?export=download&id=1YWeoFuvHeB6E4Jmt8i7fOuBrDw5d2Q9j)

## Report

[Overleaf link](https://www.overleaf.com/read/mkqjyjnntnrg)


## Related publications

### Object Recognition

#### VGG (CVPR 2014): 
* [Original paper](https://arxiv.org/abs/1409.1556)
* [Summary](docs/vgg.md)

#### ResNet (2015):
* [Original paper](https://arxiv.org/abs/1512.03385)
* [Summary](docs/resnet.md)

### Image Semantic Segmentation

#### FCN (CVPR 2015):
* [Original paper](https://arxiv.org/abs/1411.4038)
* [Summary](docs/fcn.md)

#### PSPNet (CVPR 2017):
* [Original paper](https://arxiv.org/abs/1612.01105v2)
* [Summary](docs/pspnet.md)

### Object detection

#### Faster R-CNN (NIPS 2015):
* [Original paper](https://arxiv.org/abs/1506.01497)
* [Summary](docs/fasterRCNN.md)

#### RetinaNet (ICCV 2017):
* [Original paper](https://arxiv.org/abs/1708.02002)
* [Summary](docs/retinanet.md)
