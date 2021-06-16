import lungmask
from lungmask import mask
from monai import transforms
from monai.transforms.intensity.array import ThresholdIntensity
from monai.transforms.spatial.array import Resize, Spacing
from monai.transforms.utility.dictionary import ToTensord
import torch
from tqdm import tqdm
import numpy as np
from monai.transforms import (Compose, LoadImaged, ToNumpyd, ThresholdIntensityd, AddChanneld, NormalizeIntensityd, SpatialCropd, DivisiblePadd, Spacingd, SqueezeDimd)

def mask_lung(scan_path, batch_size=20):
    model = lungmask.mask.get_model('unet', 'R231')
    device = torch.device('cuda')
    model.to(device)

    scan_dict = {
        'image' : scan_path
    }

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

def process_lung_scan(scan_dict, extremes):

    load_transformer = Compose(
        [
            LoadImaged(keys=["image"]),
            ThresholdIntensityd(keys=['image'], above = False, threshold = 1000, cval = 1000),
            ThresholdIntensityd(keys=['image'], above = True, threshold = -1024, cval = -1024),
            AddChanneld(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            SpatialCropd(keys=["image"], roi_start=(extremes[0][0], extremes[1][0], extremes[2][0]), roi_end=(extremes[0][1], extremes[1][1], extremes[2][1])),
            Spacingd(keys=["image"], pixdim=(1, 1, 1.5)),
        ]
    )

    processed_1 = load_transformer(scan_dict)

    transformer_1 = Compose(
        [
            DivisiblePadd(keys=["image"], k=16, mode='constant'),
            ToTensord(keys=['image'])
            #SqueezeDimd(keys=["image"], dim = 0),
            #ToNumpyd(keys=["image"]),
        ]
    )

    processed_2 = transformer_1(processed_1)

    affine = processed_2['image_meta_dict']['affine']

    normalized_image = processed_2['image']

    return normalized_image, affine

def preprocess(image_path):

    preprocess_dump = {}

    scan_dict = {
        'image' : image_path
    }

    im = LoadImaged(keys=['image'])(scan_dict)
    preprocess_dump['org_shape'] = im['image'].shape
    preprocess_dump['pixdim'] = im['image_meta_dict']['pixdim'][1:4]
    preprocess_dump['org_affine'] = im['image_meta_dict']['affine']

    masked_lungs = mask_lung(image_path, 5)
    right_lung_extreme = calculate_extremes(masked_lungs[0], 1)
    preprocess_dump['right_extremes'] = right_lung_extreme
    right_lung_processed = process_lung_scan(scan_dict, right_lung_extreme)

    left_lung_extreme = calculate_extremes(masked_lungs[0], 2)
    preprocess_dump['left_extremes'] = left_lung_extreme
    left_lung_processed = process_lung_scan(scan_dict, left_lung_extreme)

    
    preprocess_dump['affine'] = left_lung_processed[1]

    preprocess_dump['right_lung'] = right_lung_processed[0].unsqueeze(0)
    preprocess_dump['left_lung'] = left_lung_processed[0].unsqueeze(0)

    return preprocess_dump

def find_pad_edge(original):
    a_min = -1
    a_max = original.shape[0]

    for i in range(len(original)):
        a_min = i
        if(np.any(original[i])):
            break
    
    for i in range(len(original) - 1, 0, -1):
        a_max = i
        if(np.any(original[i])):
            break

    original = original.swapaxes(0,1)

    b_min = -1
    b_max = original.shape[0]

    for i in range(len(original)):
        b_min = i
        if(np.any(original[i])):
            break
    
    for i in range(len(original) - 1, 0, -1):
        b_max = i
        if(np.any(original[i])):
            break

    original = original.swapaxes(0,1)
    original = original.swapaxes(0,2)

    c_min = -1
    c_max = original.shape[0]

    for i in range(len(original)):
        c_min = i
        if(np.any(original[i])):
            break
    
    for i in range(len(original) - 1, 0, -1):
        c_max = i
        if(np.any(original[i])):
            break

    return a_min, a_max + 1, b_min, b_max + 1, c_min, c_max + 1


def remove_pad(mask, original):
    a_min, a_max, b_min, b_max, c_min, c_max = find_pad_edge(original)
    return mask[a_min:a_max, b_min:b_max, c_min: c_max]

def voxel_space(image, target):
    image = Resize((target[0][1]-target[0][0], target[1][1]-target[1][0], target[2][1]-target[2][0]), mode='trilinear')(np.expand_dims(image, 0))[0]
    image = ThresholdIntensity(above = False, threshold = 0.5, cval = 1)(image)
    image = ThresholdIntensity(above = True, threshold = 0.5, cval = 0)(image)

    return image

def stitch(org_shape, cropped, roi):
    holder = np.zeros(org_shape)

    holder[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1], roi[2][0]:roi[2][1]] = cropped

    return holder

def post_process(left_mask, right_mask, preprocess_dump):

    left_mask[left_mask >= 0.5] = 1
    left_mask[left_mask < 0.5] = 0

    left_mask = left_mask.astype(int)

    right_mask[right_mask >= 0.5] = 1
    right_mask[right_mask < 0.5] = 0

    right_mask = right_mask.astype(int)

    left = remove_pad(left_mask, preprocess_dump['left_lung'].squeeze(0).squeeze(0).numpy())
    right = remove_pad(right_mask, preprocess_dump['right_lung'].squeeze(0).squeeze(0).numpy())

    left = voxel_space(left, preprocess_dump['left_extremes'])
    right = voxel_space(right, preprocess_dump['right_extremes'])

    left = stitch(preprocess_dump['org_shape'], left, preprocess_dump['left_extremes'])
    right = stitch(preprocess_dump['org_shape'], right, preprocess_dump['right_extremes'])

    stitched = np.logical_or(left, right).astype(int)

    return stitched


if __name__ == "__main__":
    path = "D:\\Datasets\MSD\\Images\\lung_003.nii.gz"
    preprocess_dump = preprocess(path)

    unpad = post_process(preprocess_dump['left_lung'], preprocess_dump['right_lung'], preprocess_dump)

    import nibabel

    nimage = nibabel.Nifti1Image(unpad, preprocess_dump['org_affine'])
    nibabel.save(nimage, "D:\\Datasets\\stitched.nii.gz")
    

    
