# M5 Project: Scene Understanding for Autonomous Vehicles

The goal of this project is to learn the basic concepts and techniques to build deep neural networks to detect, segment and recognize specific objects. These techniques will be applied to environment perception for autonomous vehicles, whereby the classes of interest with regard the three tasks will be pedestrians, vehicles, road, roadside, etc.

## Team 6 members

* Alba Herrera (albaherrerapalacio at gmail dot com)
* Jorge López (jorgelopezfueyo at gmail dot com)
* Oscar Mañas (oscmansan at gmail dot com)
* Pablo Rodríguez (pablorodriper at gmail dot com)

## Instructions

```bash
python main.py [--config_file CONFIG_FILE]
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

## Goals

- [Week 2](./docs/week2.md)

## Implementation & Results

- [Week 2](./docs/week2_implementation_results.md)

## Report

[Overleaf link](https://www.overleaf.com/read/mkqjyjnntnrg)

## Slides

[Slides link](https://docs.google.com/presentation/d/1e6U8LvV8q_5QeuToiP9Zytm0M13JhbPFJgBwRQDjbQw/edit?usp=sharing)

## Model weights

[Drive link] (TODO)

## Related publications

### VGG: 
* [Original paper](https://arxiv.org/pdf/1409.1556.pdf)
* [Summary](docs/vgg.md)

### ResNet
* [Original paper](https://arxiv.org/pdf/1512.03385.pdf)
* [Summary](docs/resnet.md)
