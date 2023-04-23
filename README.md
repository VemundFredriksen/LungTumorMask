# Automatic lung tumor segmentation in CT

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)
[![Build Actions Status](https://github.com/VemundFredriksen/LungTumorMask/workflows/Build/badge.svg)](https://github.com/VemundFredriksen/LungTumorMask/actions)
[![Paper](https://zenodo.org/badge/DOI/10.1371/journal.pone.0266147.svg)](https://doi.org/10.1371/journal.pone.0266147)

This is the official repository for the paper [_"Teacher-student approach for lung tumor segmentation from mixed-supervised datasets"_](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0266147), published in PLOS ONE.

A pretrained model is made available in a command line tool and can be used as you please. However, the current model is not intended for clinical use. The model is the result of a proof-of-concept study. An improved model will be made available in the future, when more training data is made available.

![sample of masked output](https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/sample_images.png "Sample output of two different tumors")
![sample of 3d render](https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/sample_renders.png "3D render of two masked outputs")

## [Installation](https://github.com/VemundFredriksen/LungTumorMask#installation)

Software has been tested against Python `3.7-3.10`.

Stable latest release:
```
pip install https://github.com/VemundFredriksen/LungTumorMask/releases/download/v1.1.3/lungtumormask-1.1.3-py2.py3-none-any.whl
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
```

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
