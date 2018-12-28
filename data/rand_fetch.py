import sys, os
from shutil import copyfile
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, help='The source image directory.')
parser.add_argument('--output_dir', type=str, help='The target image directory.')
parser.add_argument('--copy_count', type=int, 
                    help='How many images needs to copy from source images.')
args = parser.parse_args()

if __name__ == '__main__':

    if not os.path.isdir(args.image_dir):
        print(args.image_dir, 'not exists.')
        sys.exit()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    file_list = os.listdir(args.image_dir)
    np.random.shuffle(file_list)
    file_list = file_list[:args.copy_count]

    for f in file_list:
        copyfile(os.path.join(args.image_dir, f), os.path.join(args.output_dir, f))