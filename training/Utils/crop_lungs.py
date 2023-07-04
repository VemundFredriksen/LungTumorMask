from lungmask import mask
import lungmask
import SimpleITK as sitk
import numpy as np
import nibabel
import platform
import glob
import torch
import skimage
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, ToTensord, Spacingd, DivisiblePadd, SpatialCropd, ToNumpyd, AddChanneld, SqueezeDimd, Resized, Flipd, Rotate90d, NormalizeIntensityd, ThresholdIntensityd
from Logger.loggingservice import Logger

logger = Logger("http://82.194.207.154:5000/api/log", "XqnJHdalUd")
directory_split = "\\" if platform.system() == "Windows" else "/"


def find_probable_air_value(image):
    holder = np.copy(image)
    min_val = np.amin(holder)
    holder[holder == min_val] = float('inf')
    return np.amin(holder), min_val

def mask_lung(scan_dict, batch_size=20):
    model = lungmask.mask.get_model('unet', 'R231')
    device = torch.device('cuda')
    model.to(device)

    transformer = Compose(
        [
            LoadImaged(keys=['image']),
            ToNumpyd(keys=['image']), 
        ]
    )

    scan_read = transformer(scan_dict)
    inimg_raw = scan_read['image'].swapaxes(0,2)

    tvolslices, xnew_box = lungmask.utils.preprocess(inimg_raw, resolution=[256, 256])
    tvolslices[tvolslices > 600] = 600
    tvolslices = np.divide((tvolslices + 1024), 1624)

    torch_ds_val = lungmask.utils.LungLabelsDS_inf(tvolslices)
    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, num_workers=1,
                                                 pin_memory=False)

    timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

    with torch.no_grad():
        for X in tqdm(dataloader_val):
            X = X.float().to(device)
            prediction = model(X)
            pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
            timage_res = np.vstack((timage_res, pls))

    outmask = lungmask.utils.postrocessing(timage_res)


    outmask = np.asarray(
        [lungmask.utils.reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
        dtype=np.uint8)

    outmask = np.swapaxes(outmask, 0, 2)
    #outmask = np.flip(outmask, 0)


    return outmask.astype(np.uint8), scan_read['image_meta_dict']['affine']

def segment_lung(image_path):
    sitk_image = sitk.ReadImage(image_path)
    segmentation = mask.apply(sitk_image, batch_size = 5, model = mask.get_model('unet', 'R231'))
    return segmentation

def calculate_extremes(image, annotation_value):

    holder = np.copy(image)

    x_min = float('inf')
    x_max = 0
    y_min = float('inf')
    y_max = 0
    z_min = -1
    z_max = 0

    holder[holder != annotation_value] = 0

    holder = np.swapaxes(holder, 0, 2)
    for i, layer in enumerate(holder):
        if(np.amax(layer) < 1):
            continue
        if(z_min == -1):
            z_min = i
        z_max = i

        y = np.any(layer, axis = 1)
        x = np.any(layer, axis = 0)
        y_minl, y_maxl = np.argmax(y) + 1, layer.shape[0] - np.argmax(np.flipud(y))
        x_minl, x_maxl = np.argmax(x) + 1, layer.shape[1] - np.argmax(np.flipud(x))

        if(y_minl < y_min):
            y_min = y_minl
        if(x_minl < x_min):
            x_min = x_minl
        if(y_maxl > y_max):
            y_max = y_maxl
        if(x_maxl > x_max):
            x_max = x_maxl

    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))

def process_lung_scan(scan_dict, save_directory, extremes, lung):

    load_transformer = Compose(
        [
            LoadImaged(keys=["image", "label", "boxes"]),
            ThresholdIntensityd(keys=['image'], above = False, threshold = 1000, cval = 1000),
            ThresholdIntensityd(keys=['image'], above = True, threshold = -1024, cval = -1024),
            AddChanneld(keys=["image", "label", "boxes"]),
            NormalizeIntensityd(keys=["image"]),
            SpatialCropd(keys=["image", "label", "boxes"], roi_start=(extremes[0][0], extremes[1][0], extremes[2][0]), roi_end=(extremes[0][1], extremes[1][1], extremes[2][1])),
            Spacingd(keys=["image"], pixdim=(1, 1, 1.5)),
        ]
    )

    processed_1 = load_transformer(scan_dict)
    if(np.amax(processed_1['label'][0]) == 0):
        return

    transformer_1 = Compose(
        [
            Resized(keys=["label", "boxes"], spatial_size=processed_1['image'].shape[1:]),
            ThresholdIntensityd(keys=['boxes', 'label'], above = False, threshold = 0.5, cval = 1),
            ThresholdIntensityd(keys=['boxes', 'label'], above = True, threshold = 0.5, cval = 0),
            DivisiblePadd(keys=["image", "label", "boxes"], k=16, mode='symmetric'),
            SqueezeDimd(keys=["image", "label", "boxes"], dim = 0),
            ToNumpyd(keys=["image", "label", "boxes"]),
        ]
    )

    processed_2 = transformer_1(processed_1)

    affine = processed_1['image_meta_dict']['affine']
    filename = scan_dict['image'].split(directory_split)[-1].split('.')[0]
    filename_extension = '.' + '.'.join(scan_dict['image'].split(directory_split)[-1].split('.')[1:])

    normalized_image = processed_2['image']
    image_save = nibabel.Nifti1Image(normalized_image, affine)
    boxes_save = nibabel.Nifti1Image(processed_2['boxes'], affine)
    label_save = nibabel.Nifti1Image(processed_2['label'], affine)
    nibabel.save(image_save, f"{save_directory}{directory_split}Images{directory_split}{filename}_{lung}{filename_extension}")
    nibabel.save(boxes_save, f"{save_directory}{directory_split}Boxes{directory_split}{filename}_{lung}{filename_extension}")
    nibabel.save(label_save, f"{save_directory}{directory_split}Labels{directory_split}{filename}_{lung}{filename_extension}")

passed = 0
def process_scan(scan_dict, save_directory):
    global passed
    try:
        masked, affine = mask_lung(scan_dict, batch_size=5)
    #s = nibabel.Nifti1Image(masked, affine)
    #nibabel.save(s, "D:\\Datasets\\Temp\\Images\\test4.nii.gz")
        extremes = calculate_extremes(np.copy(masked), 1)
        process_lung_scan(scan_dict, save_directory, extremes, "right")
        extremes = calculate_extremes(masked, 2)
        process_lung_scan(scan_dict, save_directory, extremes, "left")
    except:
        passed += 1
        logger.LogWarning("Skipped scan", [str(scan_dict)])
        print(f"passed {passed}")
        



def process_directory(directory, save_directory):
    for image_file in glob.glob(f"{directory}{directory_split}Images{directory_split}*.nii.gz"):
        filename = image_file.split(directory_split)[-1]

        scan_dict = {
            'image' : image_file,
            'label' : f"{directory}{directory_split}Labels{directory_split}{filename}",
            'boxes' : f"{directory}{directory_split}Boxes{directory_split}{filename}"
        }

        print(f"Processing {filename}")
        process_scan(scan_dict, save_directory)

if __name__ == "__main__":
    load_folder = "/home/tumor/data/MSD/"
    store_folder = "/home/tumor/data/MSD-Lung/"
    #load_folder = "D:\\Datasets\\Temp\\"
    #store_folder = "D:\\Datasets\\Temp\\Save\\"
    logger.LogInfo("Started cropping lungs", [])
    process_directory(load_folder, store_folder)
    logger.LogMilestone("Finished cropping lungs!", [])



