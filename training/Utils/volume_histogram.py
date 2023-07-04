import numpy as np
import nibabel
import glob
import platform
import matplotlib.pyplot as plt

directory_split = "\\" if platform.system() == "Windows" else "/"

def calculate_scan(image_path, label_path):
    
    voxel_spacing = nibabel.load(image_path).header.get_zooms()
    label_volume = nibabel.load(label_path).get_fdata()

    return calculate_volume(label_volume, voxel_spacing)

def calculate_volume(label, voxel_spacing):
    voxels = np.count_nonzero(label)
    volume = voxels * (voxel_spacing[0]/10) * (voxel_spacing[1]/10) * (voxel_spacing[2]/10)

    return volume

def calculate_all(image_directory, label_directory):
    volumes = []
    for file in glob.glob(f"{image_directory}{directory_split}*"):
        filename = file.split(directory_split)[-1]

        volumes.append(calculate_scan(file, f"{label_directory}{directory_split}{filename}"))

    return volumes

def save_volumes(volumes, save_file):
    with open(save_file, "w+") as file:
        for volume in volumes:
            file.write(f"{volume}\n")

def read_volumes(volume_file):
    file = open(volume_file, "r")
    volumes = []
    lines = file.readlines()
    for line in lines:
        volumes.append(float(line))
    
    return volumes

def plot_histogram(volumes):
    bins = [0, 5, 10, 15, 25, 50, 100, 150, 200, 300, 400, 1000]

    hist, bin_edges = np.histogram(volumes, bins)
    fig, ax = plt.subplots()

    ax.set_xticks([0.5+i for i,j in enumerate(hist)])
    ax.bar(range(len(hist)),hist,width=1,color="#15497d",edgecolor='#0b2947',align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])


    hist, bin_edges = np.histogram(volumes, bins)
    ax.bar(range(len(hist)),hist,width=1,color="#15497d",edgecolor='#0b2947',align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])

    plt.xticks(rotation = 45)
    plt.xlabel('Volume, cm\u00b3')
    plt.ylabel('Number of Tumors')
    plt.xlim(-0.5, 10.5)
    plt.show()

def plot_histograms(volumes):
    #plt.figure(figsize=(8,6))
    plt.hist(volumes[2], bins=20, alpha=0.5, label="LungDx")
    plt.hist(volumes[1], bins=20, alpha=0.5, label="Radiomics")
    plt.hist(volumes[0], bins=20, alpha=1, label="MSD")

    plt.xlabel("Volume, cm\u00b3", size=14)
    plt.ylabel("Count", size=14)
    plt.title("Tumor Volumes")
    plt.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    #vol =calculate_scan("D:\\Datasets\\Temp\\Images\\radiomics_010.nii.gz", "D:\\Datasets\\Temp\\Labels\\radiomics_010.nii.gz")
    #print(vol)
    #volumes = calculate_all("D:\\Datasets\\Temp\\Images\\", "D:\\Datasets\\Temp\\Labels\\") 
    #save_volumes(volumes, "D:\\Datasets\\radiomics.txt")
    volumes = []
    volumes.append(read_volumes("C:\\Users\\vemun\\Desktop\\msd_volumes.txt"))
    volumes.append(read_volumes("C:\\Users\\vemun\\Desktop\\radiomics_volumes.txt"))
    volumes.append(read_volumes("C:\\Users\\vemun\\Desktop\\lungdx_volumes.txt"))
    plot_histograms(volumes)