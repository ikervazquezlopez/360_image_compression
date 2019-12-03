import sys
import os
from os import listdir
from os.path import isfile, join, isdir
import cv2

original_uncompressed_dir = "uncompressed_original"
reconstucted_uncompressed_dir = "uncompressed_reconstructed"
original_tstrip_uncompressed_dir = "uncompressed_original_tstrip"
reconstructed_tstrip_uncompressed_dir = "uncompressed_reconstructed_tstrip"

original_JPEG_dir = "JPEG_original"
reconstucted_JPEG_dir = "JPEG_reconstructed"
original_tstrip_JPEG_dir = "JPEG_original_tstrip"
reconstructed_tstrip_JPEG_dir = "JPEG_reconstructed_tstrip"



original_data_dir = sys.argv[1]
reconstructed_data_dir = sys.argv[2]

test_output_dir = sys.argv[3]

if not isdir(test_output_dir):
    os.mkdir(test_output_dir)

directories = [original_uncompressed_dir, reconstucted_uncompressed_dir, original_tstrip_uncompressed_dir,
reconstructed_tstrip_uncompressed_dir, original_JPEG_dir, reconstucted_JPEG_dir, original_tstrip_JPEG_dir,
reconstructed_tstrip_JPEG_dir]

# Build directories if necessary
for d in directories:
    path = join(test_output_dir, d)
    if not isdir(path):
        os.mkdir(path)

original_filenames = [f for f in listdir(original_data_dir) if isfile(join(original_data_dir, f))]
reconstructed_filenames = [f for f in listdir(reconstructed_data_dir) if isfile(join(reconstructed_data_dir, f))]

for f in original_filenames:
    img_path = join(original_data_dir, f)

    # Save uncompressed original
    img = cv2.imread(img_path)
    path = join(test_output_dir, join(original_uncompressed_dir, f))
    cv2.imwrite(path, img)

    # Save JPEG original
    jpeg_name = f[:-4] + ".jpeg"
    path = join(test_output_dir, join(original_JPEG_dir, jpeg_name))
    cv2.imwrite(path, img)

    # TODO command to create original uncompressed tstrip
    # TODO command to create original JPEG tstrip


for f in reconstructed_filenames:
    img_path = join(reconstructed_data_dir, f)

    # Save uncompressed original
    img = cv2.imread(img_path)
    path = join(test_output_dir, join(reconstucted_uncompressed_dir, f))
    cv2.imwrite(path, img)

    # Save JPEG original
    jpeg_name = f[:-4] + ".jpeg"
    path = join(test_output_dir, join(reconstucted_JPEG_dir, jpeg_name))
    cv2.imwrite(path, img)

    # TODO command to create reconstructed uncompressed tstrip
    # TODO command to create reconstructed JPEG tstrip
