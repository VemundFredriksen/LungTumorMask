# Training code for Lung Tumor 3D Segmentation - Mixed Supervision

End to end code base for lung tumor segmentation from CT-scan using mixed supervision for deep convolutional neural network.

Takes 3D CT scans as input and outputs 3D segmentation of primary tumor.

Project aims to utilize multiple different datasets with different label types (classification labels, bounding boxes, rough segmentations, and fine segmentations).

## Installation

Requires python >= 3.6

Install python dependencies with pip. The requirements file is located in */Resources/*
```
pip install -r requirements.txt
```

Build repository files as packages using setuptools. If you alter the code, remember to run the setup again.
```
python setupy.py install
```

## Usage

**Initiate Training:**  `python .\Runable\Test\test_sevlus.py`

The training is initiated from the .\Resources\Test\test_train.yaml file that looks like this:
```
device: 'cuda'

data:
  train_dataset: "D:\\Repos\\LungTumorSegmentation\\Resources\\Test\\data_test_train.yaml"
  aug_prob: 0.1
  scale_dim:
    d_0: 32
    d_1: 32
    d_2: 32

model:
  architecture: UNet_con_double
  loss: DiceLoss
  filter_base: 64
  filter_expansion: 2
  optimizer:
    name: 'Adam'
    lr: 0.001

train:
  max_epochs: 3
  val_frequency: 1
  output_threshold: 0.5
  metric_path: "D:\\Repos\\LungTumorSegmentation\\metrics.csv"
  save_frequency: 5
  model_directory: "D:\\Repos\\LungTumorSegmentation\\models\\"
  steps_per_plot: 10
```

The paths would need to be updated.

**Plot Training Metric File**: `python interface.py plot <path_to_metric_file>`

Store the plot by specifying the store location: `python interface.py plot <path_to_metric_file> -store_file <path_to_store> -steps_per_epoch <steps_per_epoch (int)>` Example: `python interface.py plot ./metrics.csv -store_file plot.png -steps_per_epoch 120`

**Evaluate Network**: `python .\Runable\Evaluate\eval.py`

Evaluates the network on specified test images with labels. The configuration file (which here is the file at .\Resources\Windows_Files\win_eval.yaml) should contain network information as well as a reference to the data to evaluate the network on.

The evaluation config should look similar to this

```
device: 'cuda'

data:
  train_dataset: "D:\\Repos\\LungTumorSegmentation\\Resources\\Windows_Files\\evaluate_data_msd.yaml"
  scale_dim:
    d_0: 128
    d_1: 128
    d_2: 128

model:
  architecture: UNet_filter
  weights: "D:\\idun_models\\1\\model_last_128.pth"
  loss: DiceLoss
  filter_base: 128
  filter_expansion: 2
  optimizer:
    name: 'Adam'
    lr: 0.01

evaluate:
  save_segmentations: True
  save_directory: 'D:\\Repos\\LungTumorSegmentation\\m_128\\'
  output_threshold: 0.5
```

In this case, the paths would also need to be updated.

## Authors
[Vemund Fredrksen](https://github.com/VemundFredriksen), [Svein Ole M. Sevle](https://github.com/sosevle), & [André Pedersen](https://github.com/andreped).

## License
[MIT](https://choosealicense.com/licenses/mit/)
