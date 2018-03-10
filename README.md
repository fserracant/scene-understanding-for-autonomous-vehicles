# Scene understading for autonomous vehicles - Team 08: CoRN
This project is done in the framework of the Master in Computer Vision by
Universitat Autònoma de Barcelona (UAB) in the module _M5: Visual recognition_.

Abstract: "In this project we employ state-of-the-art Deep Learning algorithms for recognition, detection and segmentation tasks which are key to create a system which understands traffic scenes correctly and makes the right decision while autonomously driving".

For more information, check our report on [Overleaf][overleaf] or our presentations on [Google Drive][gdrive]. You can find summaries for some Image Classification papers on the file [_summaries_](Summaries/summaries.md). Additionally, for a quick look at what we have done in the project so far, head over to the ['The progress at a glance section'.](#the-progress-at-a-glance)
___

## Getting Started

### Prerequisites
The software need the following software to run:

- [Python 2.7][python27]
- [pip][pip-pypi]
- [Pipenv][pipenv-docs]

If you are using any Linux distribution is most probable you already have a
running version of Python installed.

As an example, if you are using an Ubuntu you can install pip and pipenv using
these commands:

```sh
apt-get install python-pip
pip install pipenv
```


### Installing
To install the virtual environment you only have to run pipenv from project's
root directory:

```sh
pipenv install
```

It is recommended to add the following environment vars to your `.bashrc`:

```sh
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
export PIPENV_MAX_DEPTH=1
```

You can find their explanation as well as more environment variables in
[configuration with environment variables][pipenv-env-vars].


## Built With
This project is using [Keras][keras] as a high-level neural networks API running
on top of [Tensorflow][tf] library.

## Run a training in the server

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python train.py -c config/tt100k_classif.py -e test -l /home/master/tmp -s /data/module5/
```

## Run in local

```bash
python code/train.py -c code/config/tt100k_classif.py -e test -l tmp -s data
```

## Pre-trained weights

<!-- TODO: add their configs! -->
You can find some the weight from our experiements in this [Google 
Drive][weights].

## The progress at a glance
### Week 1/2
#### Summary
These two weeks have been devoted to the study of state-of-the-art architectures for object recognition. In particular, we have evaluated the vainilla VGG16 architecture weights to recognise traffic signals from the TT100K dataset. We compared the performance between cropped and resized images as well as the adaption of the network to another domain (BelgiumTS). Moreover, we have trained the VGG16 both from scratch and with pre-trained weights from ImageNet on the KITTI dataset and analysed their results. Finally, we have implemented the Squeeze and Excitation Network (added to [models](code/models) and compared the vainilla ResNet50 results with its SE counterpart, both with fine-tuning and from scratch. Each dataset has been analysed to help us draw meaningful conclusions from the results obtained. Check our report and presentation for more details.
#### Results

| Model          | Train acc     | Valid acc     | Test acc      | Model         | Train acc     | Valid acc     | Test acc     |
| :-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:
| Vainilla VGG16 | 0.995         | 0.962         |  0.955        |VGG16(TT100K)* | 0.990         | 0.842         | 0.814        |
| SE-ResNet50    | 0.998         | 0.942         |  0.939        |* + resize     | 0.981         | 0.842         | 0.820        |
| VGG16(KITTI)*  | 0.888         | 0.904         |  -----        |* + crop       | 0.945         | 0.836         | 0.866        |
| * w. imagenet  | 0.978         | 0.975         |  -----        |* fine-tune BTS| 0.789         | 0.767         | 0.767        |

### Week 3/4
#### Summary
These two weeks have been devoted to the study and implementation of state-of-the-art architectures for object detection. During week 3, we have tested the vainilla YOLO (v1) architecture with the TT100K dataset (for detection, without crops of only signals) and assessed the overffiting and unbalancing problems encountered. We have disscussed about the source/s behind this problem and proposed some possible solutions (data augmentation, dropout, etc.).
Later, on week 4, ...
See our report and presentation for more details.
#### Results

### Week 5/6
#### Summary

#### Results

## Authors
- Ferran Pérez              - _dev_ - [fperezgamonal][ferran-github] - [contact](mailto:ferran.perezg@e-campus.uab.cat)
- Joan Francesc Serracant   - _dev_ - [fserracant][cesc-github] -  [contact](mailto:joanfrancesc.serracant@e-campus.uab.cat)
- Jonatan Poveda            - _dev_ - [jonpoveda][jonatan-github] - [contact](mailto:jonatan.poveda@e-campus.uab.cat)
- Martí Cobos               - _dev_ - [marticobos][marti-github] - [contact](mailto:marti.cobos@e-campus.uab.cat)

## License
This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

<!--
## Acknowledgements
-->

[python27]: https://docs.python.org/2/
[pip-pypi]: https://pypi.python.org/pypi/pip
[pipenv-docs]: http://pipenv.readthedocs.io/en/latest/
[pipenv-env-vars]: http://pipenv.readthedocs.io/en/latest/advanced/#configuration-with-environment-variables
[keras]: https://keras.io
[tf]: https://www.tensorflow.org

[ferran-github]: https://github.com/fperezgamonal
[cesc-github]: https://github.com/fserracant
[jonatan-github]: https://github.com/jonpoveda
[marti-github]: https://github.com/marticobos

[overleaf]: https://www.overleaf.com/read/rgbqdstbtmqz
[gdrive]: https://docs.google.com/presentation/d/1fmX2s14--DSvh6eTJD6e-rf5zkyVoq6O_012BAI6jJs/edit?usp=sharing
[weights]: https://drive.google.com/drive/folders/1mKUBiKQIp09UwKLrqy3C4-iG7XRnd7zZ?usp=sharing

