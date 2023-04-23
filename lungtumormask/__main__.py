import sys
import argparse
import os
from lungtumormask import mask

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
    parser.add_argument('--threshold', metavar='threshold', type=float, default=0.4, 
                        help='which threshold to use for assigning voxel-wise classes.')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    mask.mask(args.input, args.output, args.lung_filter, args.threshold)
