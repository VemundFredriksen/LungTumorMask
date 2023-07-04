# Automatic lung tumor segmentation in CT

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)
[![Build Actions Status](https://github.com/VemundFredriksen/LungTumorMask/workflows/Build/badge.svg)](https://github.com/VemundFredriksen/LungTumorMask/actions)
[![Paper](https://zenodo.org/badge/DOI/10.1371/journal.pone.0266147.svg)](https://doi.org/10.1371/journal.pone.0266147)

This is the official repository for the paper [_"Teacher-student approach for lung tumor segmentation from mixed-supervised datasets"_](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0266147), published in PLOS ONE.

A pretrained model is made available in a command line tool and can be used as you please. However, the current model is not intended for clinical use. The model is the result of a proof-of-concept study. An improved model will be made available in the future, when more training data is made available.

The source code used to train the model and conduct the experiments presented in the paper, are available in the [training](https://github.com/VemundFredriksen/LungTumorMask/tree/main/training) directory. The code is provided as is and is not meant to directly applicable to new setups, cohorts, and use cases. It should however contain all the necessary details to reproduce the experiments in the study.

<img src="https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/sample_images.png" width="70%">

<img src="https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/sample_renders.png" width="70%">

&nbsp;

## [Installation](https://github.com/VemundFredriksen/LungTumorMask#installation)

Software has been tested against Python `3.7-3.10`.

Stable latest release:
```
pip install https://github.com/VemundFredriksen/LungTumorMask/releases/download/v1.2.1/lungtumormask-1.2.1-py2.py3-none-any.whl
```

Or from source:
```
pip install git+https://github.com/VemundFredriksen/LungTumorMask
```

## [Usage](https://github.com/VemundFredriksen/LungTumorMask#usage)
After install, the software can be used as a command line tool. Simply specify the input and output filenames to run:
```
# Format
lungtumormask input_file output_file

# Example
lungtumormask patient_01.nii.gz mask_01.nii.gz

# Custom arguments
lungtumormask patient_01.nii.gz mask_01.nii.gz --lung-filter --threshold 0.3 --radius 3 --batch-size 8
```

In the last example, we filter tumor candidates outside the lungs, use a lower probability threshold to boost recall, and use a morphological smoothing step
to fill holes inside segmentations using a disk kernel of radius 3.

## [Applications](https://github.com/VemundFredriksen/LungTumorMask#applications)
* The software has been successfully integrated into the open platform [Fraxinus](https://github.com/SINTEFMedtek/Fraxinus).

## [Citation](https://github.com/VemundFredriksen/LungTumorMask#citation)
If you found this repository useful in your study, please, cite the following paper:
```
@article{fredriksen2021teacherstudent,
  title = {Teacher-student approach for lung tumor segmentation from mixed-supervised datasets},
  author = {Fredriksen, Vemund AND Sevle, Svein Ole M. AND Pedersen, André AND Langø, Thomas AND Kiss, Gabriel AND Lindseth, Frank},
  journal = {PLOS ONE},
  publisher = {Public Library of Science},
  year = {2022},
  month = {04},
  doi = {10.1371/journal.pone.0266147},
  volume = {17},
  url = {https://doi.org/10.1371/journal.pone.0266147},
  pages = {1-14},
  number = {4}
}
```
