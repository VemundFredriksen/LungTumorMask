from numpy import load
from lungtumormask.dataprocessing import preprocess, post_process
from lungtumormask.network import UNet_double
import torch as T
import nibabel

def load_model():
    if T.cuda.is_available():
        gpu_device = T.device('cuda')
    else:
        gpu_device = T.device('cpu')
    model = UNet_double(3, 1, 1, tuple([64, 128, 256, 512, 1024]), tuple([2 for i in range(4)]), num_res_units = 0)
    state_dict = T.hub.load_state_dict_from_url("https://github.com/VemundFredriksen/LungTumorMask/releases/download/0.0/dc_student.pth", progress=True, map_location=gpu_device)
    #model.load_state_dict(T.load("D:\\OneDrive\\Skole\\Universitet\\10. Semester\\Masteroppgave\\bruk_for_full_model.pth", map_location="cuda:0"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def mask(image_path, save_path):
    print("Loading model...")
    model = load_model()

    print("Preprocessing image...")
    preprocess_dump = preprocess(image_path)

    print("Looking for tumors...")
    left = model(preprocess_dump['left_lung']).squeeze(0).squeeze(0).detach().numpy()
    right = model(preprocess_dump['right_lung']).squeeze(0).squeeze(0).detach().numpy()

    print("Post-processing image...")
    infered = post_process(left, right, preprocess_dump)

    print(f"Storing segmentation at {save_path}")
    nimage = nibabel.Nifti1Image(infered, preprocess_dump['org_affine'])
    nibabel.save(nimage, save_path)