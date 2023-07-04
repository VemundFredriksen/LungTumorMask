import pydicom as dicomio
import SimpleITK as sitk
import glob
import xml.etree.ElementTree as ET
import platform
import numpy as np
import nibabel

directory_split = "\\" if platform.system() == "Windows" else "/"

def get_length(number_of_slices, thickness):
    return (number_of_slices * thickness) / 10

def convert_to_nifti(dicom_folder, save_path):
    image = []
    voxel_spacing = None
    origin = None

    for dcm_f in glob.glob(dicom_folder + f"{directory_split}*.dcm"):
        dicom = dicomio.read_file(dcm_f, force=True)

        if(dicom.Modality != 'CT' or hasattr(dicom, 'SecondaryCaptureDeviceID')):
            return False

        layer = dicom.pixel_array

        layer = layer.astype('float32')
        layer *= dicom.RescaleSlope
        layer += dicom.RescaleIntercept

        origin = dicom.ImagePositionPatient
        if(type(dicom) == dicomio.FileDataset):
            voxel_spacing = [dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]
        else:
            voxel_spacing = [dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SpacingBetweenSlices]
        
        image.append(layer)

    affine = np.zeros((4,4))
    affine[0][0] = voxel_spacing[0] * origin[0]/np.abs(origin[0])
    affine[0][-1] = origin[0] * origin[0]/np.abs(origin[0])

    affine[1][1] = voxel_spacing[1] * origin[1]/np.abs(origin[1])
    affine[1][-1] = origin[1] * origin[1]/np.abs(origin[1])

    affine[2][2] = voxel_spacing[2]
    affine[2][-1] = origin[2]
    affine[-1][-1] = 1

    image.reverse()
    nifti = nibabel.Nifti1Image(np.swapaxes(np.array(image),0,2), affine)

    nifti.update_header()
    if(get_length(len(image), voxel_spacing[-1]) > 16 and get_length(len(image), voxel_spacing[-1]) < 60):
        nibabel.save(nifti, save_path)

        return True
    return False

def generate_segbox(annotations, dicom_folder, save_path):
    image = []
    keys = annotations.keys()

    voxel_spacing = None
    origin = None

    for dcm_f in glob.glob(dicom_folder + f"{directory_split}*.dcm"):
        
        dicom = dicomio.read_file(dcm_f, force=True)
        origin = dicom.ImagePositionPatient
        if(type(dicom) == dicomio.FileDataset):
            voxel_spacing = [dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SliceThickness]
        else:
            voxel_spacing = [dicom.PixelSpacing[0], dicom.PixelSpacing[1], dicom.SpacingBetweenSlices]
        uid = dicom.SOPInstanceUID
        layer = np.zeros(dicom.pixel_array.shape)
        if(uid in keys):
            bndbox = annotations[uid]
            layer[bndbox[1] - 1:bndbox[3], bndbox[0] - 1:bndbox[2]] = 1
        image.append(layer)

    affine = np.zeros((4,4))
    affine[0][0] = voxel_spacing[0] * origin[0]/np.abs(origin[0])
    affine[0][-1] = origin[0] * origin[0]/np.abs(origin[0])

    affine[1][1] = voxel_spacing[1] * origin[1]/np.abs(origin[1])
    affine[1][-1] = origin[1] * origin[1]/np.abs(origin[1])

    affine[2][2] = voxel_spacing[2]
    affine[2][-1] = origin[2]
    affine[-1][-1] = 1

    image.reverse()
    nifti = nibabel.Nifti1Image(np.swapaxes(np.array(image),0,2), affine)

    nifti.update_header()
    if(get_length(len(image), voxel_spacing[-1]) > 20 and get_length(len(image), voxel_spacing[-1]) < 60):
        nibabel.save(nifti, save_path)

def load_annotations(folder):
    annotations = {}
    for file in glob.glob(folder + f"{directory_split}*.xml"):
        annotations[file.replace('.xml', '').split(directory_split)[-1]] = extract_boundingbox(file)

    return annotations

def extract_boundingbox(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        bndbox_xml = tree.findall('*/bndbox')[0]

        xmin = float(bndbox_xml.find('xmin').text)
        ymin = float(bndbox_xml.find('ymin').text)
        xmax = float(bndbox_xml.find('xmax').text)
        ymax = float(bndbox_xml.find('ymax').text)

        return (int(xmin), int(ymin), int(xmax), int(ymax))
    except:
        return (0, 0, 0, 0)


def convert_nsclc_petct(annotations_folder, nsclc_folder, label_folder, input_folder):
    patient_folders = glob.glob(nsclc_folder + f"{directory_split}*")
    for annotation_folder in glob.glob(annotations_folder + f"{directory_split}*"):
        patient = annotation_folder.split(directory_split)[-1]
        patient_folder = [p for p in patient_folders if patient in p][0]
        annotations = load_annotations(annotation_folder)
        
        scan_number = 1
        for examination_folder in glob.glob(patient_folder + f"{directory_split}*"):
            for scan_folder in glob.glob(examination_folder + f"{directory_split}*"):
                print(f"Converting {scan_folder}")
                if(convert_to_nifti(scan_folder, f"{input_folder}{directory_split}{patient}_scan{scan_number}.nii.gz")):
                    generate_segbox(annotations, scan_folder, f"{label_folder}{directory_split}{patient}_scan{scan_number}.nii.gz")
                    print(f"Saved {patient}_scan{scan_number}.nii.gz")
                    scan_number += 1
                
                

if __name__ == "__main__":
    #pass
    annotations = "D:\\Datasets\\NSCLC-Lung-PET-CT\\Temp\\Annotations\\"
    nsclcs_folder = "D:\\Datasets\\NSCLC-Lung-PET-CT\\Temp\Patients\\"
    labels = "D:\\Datasets\\NSCLC-Lung-PET-CT\\Temp\\Boxes\\"
    inputs = "D:\\Datasets\\NSCLC-Lung-PET-CT\\Temp\\Images\\"
    convert_nsclc_petct(annotations, nsclcs_folder, labels, inputs)