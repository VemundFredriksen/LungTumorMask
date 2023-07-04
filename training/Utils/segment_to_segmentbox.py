import nibabel
import numpy as np
import glob
import platform

directory_split = "\\" if platform.system() == "Windows" else "/"

def convert_folder(source_folder, destination_folder):
    for seg in glob.glob(source_folder + "*.nii*"):
        name = seg.split(directory_split)[-1]
        convert_file(seg, destination_folder + name)

def convert_file(source_file, destination_file):
    nifti = read_nifti(source_file)
    img = nifti.get_fdata()
    segbox = generate_segmentbox(np.array(img))
    store_nifti(nifti, segbox, destination_file)

def read_nifti(nifti_file):
    return nibabel.load(nifti_file)

def store_nifti(original, new_img, destination):
    headers = original.header
    save = nibabel.Nifti1Image(new_img, np.eye(4), headers)

    nibabel.save(save, destination)

def generate_segmentbox(image):
    image = np.swapaxes(image, 0, 2)
    for i, layer in enumerate(image):
        if(np.amax(layer) < 1):
            continue

        y = np.any(layer, axis = 1)
        x = np.any(layer, axis = 0)
        y_min, y_max = np.argmax(y) + 1, layer.shape[0] - np.argmax(np.flipud(y))
        x_min, x_max = np.argmax(x) + 1, layer.shape[1] - np.argmax(np.flipud(x))

        layer[y_min - 1:y_max, x_min - 1:x_max] = 1

    image = np.swapaxes(image, 0, 2)
    return image

if __name__ == "__main__":
    path = "D:\\Datasets\\Temp\\Labels\\"
    convert_folder(path, "D:\\Datasets\\Temp\\Boxes\\")