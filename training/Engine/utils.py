import torch as T
import yaml
import nibabel

from monai.losses import DiceLoss, GeneralizedDiceLoss, FocalLoss, TverskyLoss
from Networks.UNet4 import UNet4
from Networks.UNet5_filter_teacher import UNet5_filter_teacher
from Networks.UNet5_decoder_teacher import UNet5_decoder_teacher
from Networks.UNet_test import UNet_student
from Networks.UNet_concat_double_student import UNet_concat_student
from monai.networks.nets.unet import UNet
from Networks.monai_student import UNet_double
from Networks.monai_unet import UNet_single


def load_model(model_config, infer = False, eval = False):
    model, loss, optim = None, None, None
    model_name = model_config["architecture"]
    base = model_config["filter_base"]
    expansion = model_config["filter_expansion"]
    
    filter_layers = model_config["filters"]

    if(model_name == 'UNet_decoder'):
        model = UNet5_decoder_teacher(base, expansion)
    elif(model_name == 'UNet_filter'):
        model = UNet5_filter_teacher(base, expansion)
    elif(model_name == 'UNet_con_double'):
        model = UNet_concat_student(base, expansion)
    elif(model_name == 'UNet_monai'):
        model = UNet_single(3, 1, 1, tuple(filter_layers), tuple([2 for i in range(len(filter_layers) - 1)]))
    elif(model_name == 'UNet_monai_double'):
        model = UNet_double(3, 1, 1, tuple(filter_layers), tuple([2 for i in range(len(filter_layers) - 1)]))
    elif(model_name == 'UNet4'):
        model = UNet4(base, expansion)
    else:
        print("Architecture not found...")
        exit(1)

    if(infer):
        return model

    loss_name = model_config["loss"]
    if(loss_name == 'BCE'):
        loss = T.nn.BCELoss()
    elif(loss_name == 'DiceLoss'):
        loss = DiceLoss()
    elif(loss_name == "GenDice"):
        loss = GeneralizedDiceLoss()
    elif(loss_name == "Tversky"):
        loss = TverskyLoss(alpha=2.0, beta=10.0)
    else:
        print("Loss not found...")
        exit(1)

    if(eval):
        return model, loss

    optim_name = model_config['optimizer']['name']
    if(optim_name == 'Adam'):
        optim = T.optim.Adam(model.parameters(), model_config['optimizer']['lr'])
    else:
        print("Optimizer not found...")
        exit(1)

    return model, loss, optim

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.load(file, Loader = yaml.FullLoader)
    return config

def store_output(output, original_image, directory, affine):
    
    headers = original_image.header
    output = output.squeeze(0).squeeze(0)
    save = nibabel.Nifti1Image(output.cpu().numpy(), affine, headers)

    nibabel.save(save, directory)
