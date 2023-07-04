import torch as T
import yaml
import nibabel
import sys
from monai.losses import DiceLoss

from monai.networks.nets.unet import UNet

import numpy as np
import torch as T
from monai.data import (DataLoader, Dataset)
from monai.metrics import compute_meandice
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ToTensord,
    Resized,
    RandFlipd,
    RandRotate90d,
    ThresholdIntensityd
)

from Engine.utils import (
    load_model,
    read_yaml
)

from Networks.UNet4 import UNet4

def init_plot_file(plot_path):
    with open(plot_path, "w+") as file:
        file.write("step,loss\n")

def append_metrics(plot_path, epoch, loss):
    with open(plot_path, 'a') as file:
        file.write(f"{epoch},{loss}\n")

def main():
    path = "/cluster/work/sosevle/LungTumorSegmentation/Resources/Idun_Train_1/full_size.yaml"
    plot_path = "/cluster/work/sosevle/metrics/t_test3.csv"
    model_loc = "/cluster/work/sosevle/models/t_test3/"
    dim = (128,128,128)
    if sys.platform == "win32":
        path = "D:\\Repos\\LungTumorSegmentation\\Resources\\Test\\data_test_train.yaml"
        plot_path = "D:\\Repos\\LungTumorSegmentation\\m_test\\test.csv"
        model_loc = "D:\\Repos\\LungTumorSegmentation\\m_test\\"
        dim = (32,32,32)

    data = read_yaml(path)

    for i, d in enumerate(data['train']):
        data['train'][i]['image'] = data['image_prefix'] + d['image']
        data['train'][i]['boxes'] = data['boxes_prefix'] + d['boxes']
        data['train'][i]['label'] = data['label_prefix'] + d['label']

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "boxes", "label"]),
            ThresholdIntensityd(keys=["image"], above=False, threshold=1024, cval=1024),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            AddChanneld(keys=["image", "boxes", "label"]),
            Resized(keys=["image", "boxes", "label"], spatial_size=dim),
            RandFlipd(keys=["image", "boxes", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "boxes", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "boxes", "label"], prob=0.1, spatial_axis=2),
            RandRotate90d(keys=["image", "boxes", "label"], prob=0.1, spatial_axes=(0, 1)),
            ToTensord(keys=["image", "boxes", "label"]),
        ]
    )

    train_dataset = Dataset(data['train'], train_transform)
    train_loader = T.utils.data.DataLoader(train_dataset, 1, shuffle=True)

    device = T.device("cuda")

    #model = T.nn.Sequential(UNet(3, 1, 1, (32, 64, 128, 256), (2, 2, 2)), T.nn.Sigmoid())
    model = UNet4(32, 2)

    loss_criterion = DiceLoss()
    optimizer = T.optim.Adam(model.parameters(), 0.0005)

    model.to(device)
    model.train()

    init_plot_file(plot_path)

    optimizer.zero_grad()

    for i in range(500):
        print(f"Epoch {i}")
        epoch_loss = 0

        for j in train_loader:
            inputs, boxes, labels = j["image"].to(device), j["boxes"].to(device), j["label"].to(device)
            forwarded = model.forward(inputs)
            train_loss = loss_criterion(forwarded, labels)
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()
            print(f"loss: {train_loss.item()}")

        append_metrics(plot_path, i, epoch_loss / len(train_loader))

        if (i % 10 == 2):
            T.save(model.state_dict(), model_loc + f"m_{i}.pth")


if __name__ == "__main__":
    main()