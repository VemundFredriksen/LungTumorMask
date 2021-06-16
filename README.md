# Automatic lung tumor segmentation
This repository offers a proof of concept release of an automatic lung tumor segmentation method given a CT scan. Pre-trained weights are available for anyone to use. The repository structure is inspired by [Johannes Hofmanninger's lungmask repo](https://github.com/JoHof/lungmask), and part of the preprocessing pipeline is based on his lungmask release.

## Dependencies
In addition to the python packages specified in requirements.txt, [PyTorch](https://pytorch.org/get-started/locally/) and [Hofmanninger's lungmask](https://github.com/JoHof/lungmask) must be installed.

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
## Limitations
This repository is a proof of concept. It is not intended to be used in clinical or commercial use. However, it might be interesting for research, play a role as a baseline, or even aid semi-supervised lung tumor segmentation. 
