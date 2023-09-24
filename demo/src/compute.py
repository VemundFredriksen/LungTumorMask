def run_model(input_path):
    from lungtumormask import mask
    mask.mask(input_path, "./prediction.nii.gz", lung_filter=True, threshold=0.5, radius=1, batch_size=1)
