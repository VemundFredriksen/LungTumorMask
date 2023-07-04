import platform
import torch as T
import nibabel
import numpy as np

from monai.networks.nets.unet import UNet
from monai.data import DataLoader, Dataset
from monai.metrics import compute_meandice
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ToTensord,
    Resized,
    AsDiscrete,
    ThresholdIntensityd,
    KeepLargestConnectedComponent
)
from Engine.utils import (
    read_yaml,
    load_model,
    store_output
)

def init_eval_file(directory):
    with open(directory + "/evaluate.csv", "w+") as file:
        file.write("epoch,train_loss,val_loss,dice_score\n")

def append_eval(eval_path, input_filename, segment_filename, val_loss, dice_score):
    with open(eval_path, 'a') as file:
        file.write(f"{input_filename},{segment_filename},{val_loss},{dice_score}\n")
    
def evaluate(model, loss_function, loader, device, evaluate_config):

    model.eval()
    losses = []
    dices = []

    transform = KeepLargestConnectedComponent([1], connectivity = 3)


    with T.no_grad():
        for image in loader:
            inp, boxes, label = image["image"].to(device), image["boxes"].to(device), image["label"].to(device)
            directory_split = '\\' if platform.system() == 'Windows' else '/'
            output = model(inp)
        
            original_filename = image['image_meta_dict']['filename_or_obj'][0]
            segmentation_filename = f"{evaluate_config['save_directory']}{directory_split}seg_{original_filename.split(directory_split)[-1]}"

            original_image = nibabel.load(original_filename)
            
            loss = loss_function(output, label).item()
            
            output = output.cpu()
            output[output >= evaluate_config['output_threshold']] = 1
            output[output < evaluate_config['output_threshold']] = 0
            output = output.to(device)

            output = transform(output)

            if(evaluate_config['save_segmentations']):
                store_output(output, original_image, segmentation_filename, image['image_meta_dict']['affine'].squeeze(0).numpy())

            dice = compute_meandice(output, label).item()


            append_eval(evaluate_config['save_directory']+"evaluate.csv", original_filename, segmentation_filename, loss, dice)
            losses.append(loss)
            dices.append(dice)

    append_eval(evaluate_config['save_directory']+"evaluate.csv", 'Total', '', sum(losses)/len(losses), sum(dices)/len(dices))

def initiate(config_file):
    config = read_yaml(config_file)
    device = T.device(config["device"])

    data = read_yaml(config["data"]["train_dataset"])

    image_shape = (config["data"]["scale_dim"]["d_0"], config["data"]["scale_dim"]["d_1"], config["data"]["scale_dim"]["d_2"])

    for i, d in enumerate(data['data']):
        data['data'][i]['image'] = data['image_prefix'] + d['label']
        data['data'][i]['boxes'] = data['boxes_prefix'] + d['label']
        data['data'][i]['label'] = data['label_prefix'] + d['label']

    transform = Compose(
        [
            LoadImaged(keys=["image", "boxes", "label"]),
            AddChanneld(keys=["image", "boxes", "label"]),
            ToTensord(keys=["image", "boxes", "label"]),
        ]
    )

    dataset = Dataset(data['data'], transform)
    loader = T.utils.data.DataLoader(dataset, 1)

    model, loss = load_model(config['model'], eval = True)
    model.load_state_dict(T.load(config["model"]["weights"]))
    model.to(device)

    init_eval_file(config['evaluate']['save_directory'])

    evaluate(model, loss, loader, device, config['evaluate'])
