import sys
import argparse
import os

def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='input', type=path, help='Path to the input image, should be .nifti')
    parser.add_argument('output', metavar='output', type=str, help='Filepath for output tumormask')
    parser.add_argument('--lung-filter', action='store_true', help='whether to apply lungmask postprocessing.')
    parser.add_argument('--threshold', metavar='threshold', type=float, default=0.5, 
                        help='which threshold to use for assigning voxel-wise classes.')
    parser.add_argument('--radius', metavar='radius', type=int, default=1,
                        help='which radius to use for morphological post-processing segmentation smoothing.')
    parser.add_argument('--batch-size', metavar='batch-size', type=int, default=5,
                        help='which batch size to use for lungmask inference.')
    parser.add_argument('--cpu', action='store_true', help='whether to force computation to happen on CPU only.')

    args = parser.parse_args(sys.argv[1:])

    # whether to force CPU computation
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # import method here to enable faster testing
    from lungtumormask import mask
    
    mask.mask(args.input, args.output, args.lung_filter, args.threshold, args.radius, args.batch_size)
