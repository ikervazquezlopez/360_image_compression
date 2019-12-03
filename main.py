import sys
from os import listdir
from os.path import isfile, join, isdir
import os


result_dir_name = "results"
comp_dir_name = "compressed"




if __name__ == "__main__":
    directory = sys.argv[1]
    filenames = [f for f in listdir(directory) if isfile(join(directory, f))]

    if not isdir(result_dir_name):
        os.mkdir(result_dir_name)
    if not isdir(comp_dir_name):
        os.mkdir(comp_dir_name)

    for f in filenames:
        in_path = join(directory, f)
        comp_path = join(comp_dir_name, f)
        out_path = join(result_dir_name, f)

        os.system("python 360_image_compression.py {} {}".format(in_path, comp_path))
