import SimpleITK as sitk
import glob
import os
import platform

directory_split = "\\" if platform.system() == "Windows" else '/'

def read_itk(filename):
    return sitk.ReadImage(filename)

def read_dicoms(dicom_folder):
    reader = sitk.ImageSeriesReader()
    dicoms = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicoms)

    return reader.Execute()

def write_nrrd(nrrd, filename):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(nrrd)

def itk_to_nrrd(source, destination):
    queue = glob.glob(source)
    for f in queue:
        if(os.path.isdir(f)):
            os.mkdir(destination + f.split(directory_split)[-1])
            itk_to_nrrd(f + "\\*", destination + f.split(directory_split)[-1] + "\\")
        elif(os.path.isfile(f)):
            itk = read_itk(f)
            dst = destination + "\\" + f.split(directory_split)[-1]
            dst = dst[:dst.rfind('.')] + ".nrrd"
            write_nrrd(itk, dst)