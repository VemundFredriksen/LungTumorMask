import glob
import platform
import nibabel
import os
import numpy as np
import torch
from monai.transforms import Compose, Spacingd, ToTensord, LoadImaged, AddChanneld, ThresholdIntensityd, LoadImage, ScaleIntensityRanged, NormalizeIntensityd
from monai.data import NiftiSaver
from segment_to_segmentbox import generate_segmentbox
import pickle
from Logger.loggingservice import Logger

logger = Logger("http://82.194.207.154:5000/api/log", "XqnJHdalUd")

directory_split = "\\" if platform.system() == "Windows" else "/"

voxel_normalizer = Compose(
        [
            LoadImaged(keys=['image', 'boxes', 'label']),
            ThresholdIntensityd(keys=['image'], above = False, threshold = 1000, cval = 1000),
            ThresholdIntensityd(keys=['image'], above = True, threshold = -1024, cval = -1024),
            AddChanneld(keys=['image', 'boxes', 'label']),
            Spacingd(keys=['image', 'boxes', 'label'], pixdim=(0.7, 0.7, 0.7)),
            #ScaleIntensityRanged(keys=['image'], a_min=-1024, a_max=400, b_min=-1, b_max=1),
            NormalizeIntensityd(keys=['image']),
            ThresholdIntensityd(keys=['boxes'], above = False, threshold = 0.5, cval = 1),
            ThresholdIntensityd(keys=['boxes'], above = True, threshold = 0.5, cval = 0),
            ToTensord(keys=['image', 'boxes', 'label'])
        ]
    )

# voxel_normalizer = Compose(
#     [
#         LoadImaged(keys=['label']),
#         AddChanneld(keys=['label']),
#         Spacingd(keys=['label'], pixdim=(0.7, 0.7, 0.7)),
#         ToTensord(keys=['label'])
#     ]
# )


def crop(image, center, dimensions):
    #holder = np.zeros(shape=dimensions)
    holder = np.full(dimensions, -1).astype(np.float32)
    
    crop = image[(center[0]-int(dimensions[0]/2)):center[0]+int(dimensions[0]/2), 
        (center[1]-int(dimensions[1]/2)):center[1]+int(dimensions[1]/2), 
        (center[2]-int(dimensions[2]/2)):center[2]+int(dimensions[2]/2)]

    #return crop
    holder[:crop.shape[0], :crop.shape[1], : crop.shape[2]] = crop
    return holder

def find_center(box_array):
    
    z_min = -1
    z_max = 0
    y_min = float('inf')
    y_max = 0
    x_min = float('inf')
    x_max = 0


    box_array = box_array.swapaxes(0, 2)
    for i, layer in enumerate(box_array):
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
    
    return x_min + int((x_max - x_min)/2), y_min + int((y_max - y_min)/2), z_min + int((z_max - z_min)/2)

def crop_and_store(root_folder, save_folder):    
    #os.mkdir(f"{save_folder}{directory_split}Images")
    #os.mkdir(f"{save_folder}{directory_split}Boxes")
    #os.mkdir(f"{save_folder}{directory_split}Labels")

    image_directory = f'{root_folder}{directory_split}Images{directory_split}'
    box_directory = f'{root_folder}{directory_split}Boxes{directory_split}'
    label_directory = f'{root_folder}{directory_split}Labels{directory_split}'

    crop_corner_dict = {}

    for box_file in glob.glob(f'{box_directory}*'):
        file_name = box_file.split(directory_split)[-1]
        print(f"Processing {file_name}")

        data = [{'image' : f"{image_directory}{file_name}", 'boxes' : f"{box_directory}{file_name}", 'label': f"{label_directory}{file_name}"}]

        #imloader = LoadImage()
        #org_box_shape = imloader(data[0]['boxes'])[0].shape

        data = voxel_normalizer(data)

        
        box = nibabel.load(f"{box_directory}{file_name}")
        new_box_arr = torch.nn.functional.interpolate(torch.Tensor(box.get_fdata()).unsqueeze(0).unsqueeze(0), 
            data[0]['image'].numpy().shape[1:], mode='trilinear', align_corners = False).squeeze(0).squeeze(0)
        new_box_arr[new_box_arr < 0.5] = 0
        new_box_arr[new_box_arr >= 0.5] = 1
        #new_box_arr =

        #new_box_arr = data[0]['boxes'].squeeze()
        try:
            center = find_center(np.copy(new_box_arr.numpy()))
        except:
            print(f"Couldnt calculate center of {file_name}")
            continue

    
        centered_image = crop(data[0]['image'].squeeze(0).numpy(), center, (128, 128, 128))
        #centered_box = crop(data[0]['boxes'].squeeze(0).numpy(), center, (128, 128, 128))
        centered_box = crop(new_box_arr, center, (128, 128, 128))
        centered_label = crop(data[0]['label'].squeeze(0).numpy(), center, (128, 128, 128))
        centered_label[centered_label >= 0.5] = 1
        centered_label[centered_label < 0.5] = 0
        #centered_box = generate_segmentbox(np.copy(centered_label))
        #centered_box[centered_box < 0.5] = 0
        #centered_box[centered_box > 0.5] = 1


        affine = data[0]['image_meta_dict']['affine']

        box_nifti = nibabel.Nifti1Image(centered_box, affine)
        image_nifti = nibabel.Nifti1Image(centered_image, affine)
        label_nifti = nibabel.Nifti1Image(centered_label, affine)

        nibabel.save(box_nifti, f"{save_folder}{directory_split}Boxes{directory_split}{file_name}")
        nibabel.save(image_nifti, f"{save_folder}{directory_split}Images{directory_split}{file_name}")
        nibabel.save(label_nifti, f"{save_folder}{directory_split}Labels{directory_split}{file_name}")

        org_box_shape = box.get_fdata().shape
        crop_corner_dict[file_name] = (center, data[0]['image'].squeeze(0).numpy().shape, org_box_shape)
        print(f"Saving {file_name}")
    

    pickle.dump(crop_corner_dict, open( f"{save_folder}{directory_split}crop_reference.bin", "wb" ))

def inject_image(holder, image, corner):

    x_max = min(corner[0]+image.shape[0], holder.shape[0])
    y_max = min(corner[1]+image.shape[1], holder.shape[1])
    z_max = min(corner[2]+image.shape[2], holder.shape[2])

    x_imslice = corner[0]+image.shape[0] - x_max
    y_imslice = corner[1]+image.shape[1] - y_max
    z_imslice = corner[2]+image.shape[2] - z_max

    holder[corner[0]:x_max, corner[1]:y_max, corner[2]: z_max] = image[:image.shape[0] - x_imslice, :image.shape[1] - y_imslice, :image.shape[2] - z_imslice]
    return holder

def calculate_corner(imputation_shape, center_in_holder):
    x = center_in_holder[0] - imputation_shape[0]/2
    y = center_in_holder[1] - imputation_shape[1]/2
    z = center_in_holder[2] - imputation_shape[2]/2
    return (int(x), int(y), int(z))

def load_and_relocate(reference_file, segment_folder, store_folder):

    ref_dict = pickle.load(open(reference_file, 'rb'))
    
    logger.LogInfo("Starting load and relocate", [])
    for image_file in glob.glob(f"{segment_folder}{directory_split}*.nii.gz"):
        try:        
            file_name = image_file.split(directory_split)[-1]
            references = ref_dict[file_name[4:]]

            holder = np.zeros(shape=references[1])
            nib = nibabel.load(image_file)
            image = nib.get_fdata()
            corner = calculate_corner(image.shape, references[0])
            injected = inject_image(holder, image, corner)

            injected = torch.nn.functional.interpolate(torch.Tensor(injected).unsqueeze(0).unsqueeze(0), size=references[-1], mode='trilinear').squeeze(0).squeeze(0)
            injected[injected >= 0.5] = 1
            injected[injected < 0.5] = 0
            injected = injected.numpy()
            nifti = nibabel.Nifti1Image(injected, nib.affine)
            nibabel.save(nifti, f"{save_folder}{directory_split}{file_name}")

            print(file_name)
        except:
            logger.LogWarning(f"Failed relocate", [image_file])



if __name__ == "__main__":

    crop_and_store("D:\\Datasets\\Temp\\", "D:\\Datasets\\Temp\\Save\\")

    #ref = "/home/tumor/data/LungDx-Tumor/crop_reference.bin"
    #segment_folder = "/home/tumor/data/LungDx-Tumor/Labels/"
    #save_folder = "/home/tumor/data/LungDx-Full/Labels/"

    #load_and_relocate(ref, segment_folder, save_folder)
    #logger.LogMilestone("Finished relocating", [])
