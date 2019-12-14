import sys
from os import listdir
from os.path import isfile, join, isdir
import os


result_dir_name = "results"
comp_dir_name = "compressed"

comp_dir_png = join(comp_dir_name, "PNG")
comp_dir_jpeg = join(comp_dir_name, "JPEG")

result_dir_png = join(result_dir_name, "PNG")
result_dir_jpeg = join(result_dir_name, "JPEG")



if __name__ == "__main__":
    directory = sys.argv[1]
    filenames = [f for f in listdir(directory) if isfile(join(directory, f) and ".tif" in f)]

    if not isdir(result_dir_name):
        os.mkdir(result_dir_name)
    if not isdir(comp_dir_name):
        os.mkdir(comp_dir_name)
    if not isdir(comp_dir_png):
        os.mkdir(comp_dir_png)
    if not isdir(comp_dir_jpeg):
        os.mkdir(comp_dir_jpeg)
    if not isdir(result_dir_png):
        os.mkdir(result_dir_png)
    if not isdir(result_dir_jpeg):
        os.mkdir(result_dir_jpeg)

    for f in filenames:
        in_path = join(directory, f)
        name = f.split('.')
        print("== {} ===============".format(name))

        # COMPRESSION OF IMAGES
        comp_path_png = join(comp_dir_png, name[0] + ".png")
        comp_path_jpeg = join(comp_dir_jpeg, name[0] + ".jpeg")

        print("Compression PNG...")
        os.system("python 360_image_compression.py {} {}".format(in_path, comp_path_png))
        print("Compression JPEG...")
        os.system("python 360_image_compression.py {} {}".format(in_path, comp_path_jpeg))

        # DECOMPRESSION OF IMAGES
        out_path_png = join(result_dir_png, name[0] + ".png")
        out_path_jpeg = join(result_dir_jpeg, name[0] + ".png")

        print("Decompression PNG...")
        os.system("python 360_image_decompression.py {} {}".format(comp_path_png, out_path_png))
        print("Decompression from JPEG...")
        os.system("python 360_image_decompression.py {} {}".format(comp_path_jpeg, out_path_jpeg))


    os.system("python create_test_files.py data\ results\ testing\ ")
