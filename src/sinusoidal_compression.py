import numpy as np
import math
import cv2
import sinusoidal_downsampling as sd
import sys
import os
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm


equ_dirname = "equirectangular"
sin_dirname = "sinusoidal"
comp_dirname = "compressed"
decomp_dirname = "decompressed"


if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # Build directories to store the output files
    if not isdir(out_dir):
        os.mkdir(out_dir)
    if not isdir(join(out_dir, equ_dirname)):
        os.mkdir(join(out_dir, equ_dirname))
    if not isdir(join(out_dir, sin_dirname)):
        os.mkdir(join(out_dir, sin_dirname))
    if not isdir(join(out_dir, comp_dirname)):
        os.mkdir(join(out_dir, comp_dirname))
    if not isdir(join(out_dir, decomp_dirname)):
        os.mkdir(join(out_dir, decomp_dirname))



    filenames = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    for f in tqdm(filenames):
        equ = cv2.imread(join(in_dir, f), cv2.IMREAD_COLOR)
        sin = sd.sinusoidal_downsampling(equ)
        comp = sd.sinusoidal_compression(equ)
        decomp = sd.sinusoidal_decompression(comp)

        # Save files in corressponding directories
        cv2.imwrite(join(join(out_dir, equ_dirname), f), equ)
        cv2.imwrite(join(join(out_dir, sin_dirname), f), sin)
        cv2.imwrite(join(join(out_dir, comp_dirname), f), comp)
        cv2.imwrite(join(join(out_dir, decomp_dirname), f), decomp)
