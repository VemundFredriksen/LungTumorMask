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
    ThresholdIntensityd,
    ScaleIntensityRanged
)

from Engine.utils import (
    load_model,
    read_yaml
)


from Engine.utils import (
    load_model,
    read_yaml
)

from Engine.utils import (
    load_model,
    read_yaml
)


def init_plot_file(plot_path):
    with open(plot_path, "w+") as file:
        file.write("step,train_loss,val_loss,dice_score\n")

def append_metrics(plot_path, epoch, train_loss, val_loss, dice_score):
    with open(plot_path, 'a') as file:
        file.write(f"{epoch},{train_loss},{val_loss},{dice_score}\n")

def train(model, loss_function, optimizer, train_loader, val_loader, device, train_config):
    best_val_loss = float('inf')
    val_loss = float('inf')
    total_steps = 0
    dcs_metric = 0
    step_loss = 0

    optimizer.zero_grad()

    for epoch in range(train_config['max_epochs']):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{train_config['max_epochs']}")
        model.train()
        train_loss = 0
        step = 0
        accumulated_loss = 0

        if((epoch + 1) % train_config['save_frequency'] == 0):
            T.save(model.state_dict(), train_config['model_directory'] + f"model_{epoch + 1}.pth")

        for batch_data in train_loader:
            total_steps += 1
            step += 1
            inputs, boxes, labels = batch_data["image"].to(device), batch_data["boxes"].to(device), batch_data["label"].to(device)
            loss = model.train_step(inputs, labels, loss_function, boxes)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
            step_loss += loss.detach().item()
            
            print(
                f"step {step}/{len(train_loader.dataset) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.6f}")
            epoch_len = len(train_loader.dataset) // train_loader.batch_size
            accumulated_loss += loss

            if(total_steps % train_config['batch_size'] == 0):

                accumulated_loss = 0
                step_loss /= train_config['batch_size']
                append_metrics(train_config['metric_path'], total_steps, step_loss, val_loss, dcs_metric)
                step_loss = 0

        train_loss /= len(train_loader.dataset)
        print(f"epoch {epoch + 1} average loss: {train_loss:.6f}")

        

        if((epoch + 1) % train_config['val_frequency'] == 0):
            model.eval()


            print("Start eval")

            step = 0
            val_loss = 0
            dcs_metric = 0

            with T.no_grad():
                for batch_data in val_loader:
                    step += 1
                    inputs, boxes, labels = batch_data["image"].to(device), batch_data["boxes"].to(device), batch_data["label"].to(device)
                    outputs = model(inputs)

                    loss = loss_function(outputs, labels)

                    outputs = outputs.cpu()

                    outputs[outputs >= train_config['output_threshold']] = 1
                    outputs[outputs < train_config['output_threshold']] = 0
                    outputs = outputs.to(device)

                    dcs_metric += T.mean(compute_meandice(outputs.unsqueeze(0), labels.unsqueeze(0))).item()
                    val_loss += loss.item()
                    
            dcs_metric /= len(val_loader.dataset)

                
            val_loss /= step

            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                T.save(model.state_dict(), train_config['model_directory'] + f"model_best.pth")

            print(f"epoch {epoch + 1} validation loss: {val_loss:.6f}")
            print(f"epoch {epoch + 1} validation dice score: {dcs_metric:.6f}")

        
        T.save(model.state_dict(), train_config['model_directory'] + f"model_last.pth")

def initiate(config_path):
    config = read_yaml(config_path)
    init_plot_file(config['train']['metric_path'])
    data_paths = config["data"]["train_dataset"]
    image_shape = (config["data"]["scale_dim"]["d_0"], config["data"]["scale_dim"]["d_1"], config["data"]["scale_dim"]["d_2"])

    combined_train = []
    combined_val = []

    for data in data_paths:
        prefixes = read_yaml(data)

        for i, d in enumerate(read_yaml(data)['train']):
            instance = {
                'image' : prefixes['image_prefix'] + d['label'],
                'label' : prefixes['label_prefix'] + d['label'],
                'boxes' : prefixes['boxes_prefix'] + d['label']
            }
            combined_train.append(instance)
    
    for data in data_paths:
        prefixes = read_yaml(data)

        for i, d in enumerate(read_yaml(data)['train']):
            instance = {
                'image' : prefixes['image_prefix'] + d['label'],
                'label' : prefixes['label_prefix'] + d['label'],
                'boxes' : prefixes['boxes_prefix'] + d['label']
            }
            combined_val.append(instance)

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "boxes", "label"]),
            AddChanneld(keys=["image", "boxes", "label"]),
            RandFlipd(keys=["image", "boxes", "label"], prob=config["data"]["aug_prob"], spatial_axis=0),
            RandFlipd(keys=["image", "boxes", "label"], prob=config["data"]["aug_prob"], spatial_axis=1),
            RandFlipd(keys=["image", "boxes", "label"], prob=config["data"]["aug_prob"], spatial_axis=2),
            RandRotate90d(keys=["image", "boxes", "label"], prob=config["data"]["aug_prob"], spatial_axes=(0, 1)),
            RandRotate90d(keys=["image", "boxes", "label"], prob=config["data"]["aug_prob"], spatial_axes=(0, 2)),
            RandRotate90d(keys=["image", "boxes", "label"], prob=config["data"]["aug_prob"], spatial_axes=(1, 2)),
            ToTensord(keys=["image", "boxes", "label"]),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "boxes", "label"]),
            AddChanneld(keys=["image", "boxes", "label"]),
            ToTensord(keys=["image", "boxes", "label"]),
        ]
    )

    train_dataset = Dataset(combined_train, train_transform)
    train_loader = T.utils.data.DataLoader(train_dataset, 1, shuffle = True)

    val_dataset = Dataset(combined_val, val_transform)
    val_loader = T.utils.data.DataLoader(val_dataset, 1)

    device = T.device(config["device"])

    model, loss, optimizer = load_model(config['model'])
    if (config["model"]["weights"]):
        model.load_state_dict(T.load(config["model"]["weights"]))
    model.to(device)

    print("initiates training!")
    train(model, loss, optimizer, train_loader, val_loader, device, config['train'])
