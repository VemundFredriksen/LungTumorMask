import glob
import platform

directory_split = "\\" if platform.system() == "Windows" else "/"


def generate_training_data(save_location, msd_directory, target_platform):
    target_split = "\\\\" if target_platform.lower() == "windows" else "/"
    with open(save_location, 'w+') as f:
        print(msd_directory)
        str = "image_prefix: \"\"\nlabel_prefix: \"\"\ntrain:\n\t[\n".replace("\t", "  ")
        f.write(str)
        for file in sorted(glob.glob(msd_directory+f"{directory_split}Images{directory_split}*")):
            print(file)
            name = file.split('\\')[-1]
            f.write("\t\t{\n".replace("\t", "  ") + f"\t\t\t'image' : '{file.split(directory_split)[-1]}', \n\t\t\t'label': '{file.split(directory_split)[-1]}', \n\t\t\t'boxes': '{file.split(directory_split)[-1]}'".replace("\t", "  ") + "\n\t\t},\n".replace("\t", "  "))

        f.write('\n]')

        f.write("\nval : []")
            



if __name__ == "__main__":
    path = "/home/tumor/data/LungDx-Lung/"

    generate_training_data(path + "/lungdx_dataset.yaml", path, "linux")
