from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
from glob import glob
import os
import platform

directory_split = "\\" if platform.system() == "Windows" else "/"

def format_scan(dicom_path, rstruct_path, savepath):
    dcmrtstruct2nii(rstruct_path, dicom_path, savepath, mask_foreground_value=1, structures=['GTV-1'])

def move_temporary_file(save_directory, file_name):
    try:
        os.rename(f"{save_directory}{directory_split}temp{directory_split}mask_GTV-1.nii.gz", f"{save_directory}{directory_split}Labels{directory_split}{file_name}")
        os.rename(f"{save_directory}{directory_split}temp{directory_split}image.nii.gz", f"{save_directory}{directory_split}Images{directory_split}{file_name}")
    except:
        print("Segmentation or raw image not found\nmoving on...")

def format_radiomics(load_directory, save_directory):

    os.mkdir(f"{save_directory}{directory_split}temp")
    os.mkdir(f"{save_directory}{directory_split}Images")
    os.mkdir(f"{save_directory}{directory_split}Labels")
    temp_dir = f"{save_directory}{directory_split}temp"

    for scan in sorted(glob(f"{load_directory}{directory_split}*")):
        file_name = "radiomics_" + scan.split(directory_split)[-1].split("-")[-1]
        scan_folder = glob(f"{scan}{directory_split}*")[0]
        components = glob(f"{scan_folder}{directory_split}*")

        dcm = components[0]
        rstruct = components[0]

        for component in components:
            if(len(glob(f"{component}{directory_split}*")) > 1):
                dcm = component
                break

        for component in components:
            rs = glob(f"{component}{directory_split}*")[0]
            try:
                list_rt_structs(rs)
                rstruct = rs
            except:
                pass

        format_scan(dcm, rstruct, temp_dir)
        move_temporary_file(save_directory, f"{file_name}.nii.gz")



if __name__ == "__main__":
    format_radiomics("D:\\Datasets\\NSCLC-Radiomics\\NSCLC-Radiomics\\", "D:\\Datasets\\Temp\\")