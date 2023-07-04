import platform
import nibabel
import torch as T
from monai.data import DataLoader, Dataset
from monai.transforms import (
    AddChanneld, 
    Compose, 
    LoadImaged,
    NormalizeIntensityd, 
    Resized,
    ThresholdIntensityd, 
    ToTensord
)

from Engine.utils import (
    load_model,
    read_yaml,
    store_output
)
    
def infer(model, loader, device, infer_config):

    model.eval()

    with T.no_grad():
        for image in loader:
            inp, boxes = image["image"].to(device), image["boxes"].to(device)
            directory_split = '\\' if platform.system() == 'Windows' else '/'
            output = model(inp)
        
            original_filename = image['image_meta_dict']['filename_or_obj'][0]
            segmentation_filename = f"{infer_config['save_directory']}{directory_split}seg_{original_filename.split(directory_split)[-1]}"

            original_image = nibabel.load(original_filename)
            upsample = T.nn.Upsample(original_image.shape, mode='trilinear', align_corners = False)
            output = upsample(output)

            print(image['image_meta_dict']['affine'])
            store_output(output, original_image, segmentation_filename, image['image_meta_dict']['affine'].squeeze(0).numpy())


def initiate(config_file):
    config = read_yaml(config_file)
    device = T.device(config["device"])

    data = read_yaml(config["data"]["dataset"])

    image_shape = (config["data"]["scale_dim"]["d_0"], config["data"]["scale_dim"]["d_1"], config["data"]["scale_dim"]["d_2"])

    for i, d in enumerate(data['test']):
        data['test'][i]['image'] = data['image_prefix'] + d['image']
        data['test'][i]['boxes'] = data['boxes_prefix'] + d['boxes']


    transform = Compose(
        [
            LoadImaged(keys=["image", "boxes"]),
            AddChanneld(keys=["image", "boxes"]),
            ToTensord(keys=["image", "boxes"]),
        ]
    )

    dataset = Dataset(data['data'], transform)
    loader = T.utils.data.DataLoader(dataset, 1)

    model = load_model(config['model'], infer = True)
    model.load_state_dict(T.load(config["model"]["weights"]))
    model.to(device)

    logger.LogInfo("Starting inference!", [str(data)])
    infer(model, loader, device, config['inference'])
    logger.LogMilestone("Inference finished!", [])
