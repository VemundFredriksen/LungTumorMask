# Automatic lung tumor segmentation in CT
This is the official repository for the paper [_"Teacher-Student Architecture for Mixed Supervised Lung Tumor Segmentation"_](https://arxiv.org/abs/2112.11541), submitted to the International Journal of Computer Assisted Radiology and Surgery ([IJCARS](https://www.springer.com/journal/11548)).

A pretrained model is made available and can be used as you please. However, the current model is not intended for clinical use. The model is a result of a proof-of-concept study, and an improved model will be made available in the near future, when more training data is made available.

![sample of masked output](https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/sample_images.png "Sample output of two different tumors")
![sample of 3d render](https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0.1/sample_renders.png "3D render of two masked outputs")

## Dependencies
In addition to the python packages specified in requirements.txt, [PyTorch](https://pytorch.org/get-started/locally/) and [lungmask](https://github.com/JoHof/lungmask) must be installed.

## Installation
```
pip install git+https://github.com/VemundFredriksen/LungTumorMask
```

## Usage
When the package is installed through pip, simply specify the input and output filenames.
```
# Format
lungtumormask input_file output_file

# Example
lungtumormask patient_01.nii.gz mask_01.nii.gz
```

## Acknowledgements
If you found this repository useful in your study, please, cite the following paper:
```
@misc{fredriksen2021teacherstudent,
title={Teacher-Student Architecture for Mixed Supervised Lung Tumor Segmentation}, 
author={Vemund Fredriksen and Svein Ole M. Svele and André Pedersen and Thomas Langø and Gabriel Kiss and Frank Lindseth},
year={2021},
eprint={2112.11541},
archivePrefix={arXiv},
primaryClass={eess.IV}}
```
