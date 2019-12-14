import cv2
import numpy as np
import math
import skimage.measure as metrics
import cupy as cp
import sys
from os import listdir
from os.path import isfile, join, isdir
import os


"""

def show_image(img):
	cv2.imshow("test", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def SSIM(original, comp):
	print("SSIM")
	# Constant initializations
	k1 = 0.01
	k2 = 0.03
	L = 2**8 - 1
	c1 = (k1 * L) * (k1 * L)
	c2 = (k2 * L) * (k2 * L)

	# Values computed from the images
	print("Loading image 1")
	img0_Y = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)[:,:,0]
	h, w = img0_Y.shape
	del original
	img0_Y = cp.array(img0_Y)
	print("Loading image 2")
	img1_Y = cv2.cvtColor(comp, cv2.COLOR_BGR2YCR_CB)[:,:,0]
	del comp
	img1_Y = cp.array(img1_Y)

	print("Computing means")
	mux = cp.mean(img0_Y)
	muy = cp.mean(img1_Y)
	print(muy)
	print("Computing variances")
	sigma2x = cp.var(img0_Y)
	sigma2y = cp.var(img1_Y)
	print(sigma2x)
	print("Computing covariance")
	sigmaxy = cp.sum((img0_Y - mux)*(img1_Y - muy)) / (h*w)
	#sigmaxy = cp.cov(img0_Y, img1_Y)
	#sigmaxy = cp.asnumpy(sigmaxy)
	print(sigmaxy)

	# Compute SSMI
	print("Computing SSIM")
	numerator = (2*mux*muy + c1) * (2*sigmaxy + c2)
	denominator = (mux*mux + muy*muy + c1) * (sigma2x*sigma2x + sigma2y*sigma2y + c2)
	ssim = numerator / denominator
	return ssim
"""


def compute_ssim(img0, img1):
	img0 = cv2.resize(img0, None, fx=0.5, fy=0.5)
	img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
	return metrics.compare_ssim(img0, img1, multichannel=True)

"""
original = cv2.imread('testing/JPEG_original/original.jpeg')
h, w, c = original.shape
original = original[:h, :w-2]
original = cv2.resize(original, None, fx=0.5, fy=0.5)
print(original.shape)
comp = cv2.imread('testing/JPEG_reconstructed/reconstructed.jpeg')
comp = cv2.resize(comp, None, fx=0.5, fy=0.5)
print(comp.shape)
#comp = cv2.imread('data/test.jpeg')
h, w, c = comp.shape
#comp = comp[:h, :w]
#comp = comp[:h, :w]


#print(PSNR(original, comp))
print(cv2.PSNR(original, comp))
#print(SSIM(original,comp))
print(metrics.compare_ssim(original, comp, multichannel=True))

#diff = cv2.subtract(original, comp)
#diff = diff * 2
#cv2.imshow("test", diff)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
"""


original_uncompressed_dir = "uncompressed_original"
reconstucted_uncompressed_dir = "uncompressed_reconstructed"
original_tstrip_uncompressed_dir = "uncompressed_original_tstrip"
reconstructed_tstrip_uncompressed_dir = "uncompressed_reconstructed_tstrip"

original_JPEG_dir = "JPEG_original"
reconstucted_JPEG_dir = "JPEG_reconstructed"
original_tstrip_JPEG_dir = "JPEG_original_tstrip"
reconstructed_tstrip_JPEG_dir = "JPEG_reconstructed_tstrip"


if __name__ == "__main__":
	print("=============")
	print("IMPORTANT NOTE: metrics are computed over shrinked images to fit images in memory and make it run!!")
	print("=============")

	test_dir = sys.argv[1]
	source_filename_dir = join(test_dir, original_uncompressed_dir)
	filenames = [f for f in listdir(source_filename_dir) if isfile(join(source_filename_dir, f))]

	for f in filenames:
		print("+++++++++++ {} +++++++++".format(f))
		
		# Original PNG vs Reconstructed from PNG
		print("== Original PNG vs Reconstructed from PNG ==")
		original_img_png = cv2.imread(join(source_filename_dir, f))
		h, w, _ = original_img_png.shape
		original_img_png = original_img_png[:h, :w-2]
		rec_image_png = cv2.imread(join(join(test_dir, reconstucted_uncompressed_dir), f))
		print(compute_ssim(original_img_png, rec_image_png))
		del rec_image_png

		# Original PNG vs Reconstructed from JPEG
		print("== Original PNG vs Reconstructed from JPEG ==")
		#original_img_png = cv2.imread(join(source_filename_dir, f))
		#h, w, _ = original_img_png.shape
		#original_img_png = original_img_png[:h, :w-2]
		rec_image_jpeg = cv2.imread(join(join(test_dir, reconstucted_JPEG_dir), f))
		print(compute_ssim(original_img_png, rec_image_jpeg))
		del original_img_png

		# Original JPEG vs Reconstruced from JPEG
		print("== Original JPEG vs Reconstructed from JPEG ==")
		jpeg_name = f.split('.')[0] + ".jpeg"
		original_img_jpeg = cv2.imread(join(join(test_dir, original_JPEG_dir), jpeg_name))
		h, w, _ = original_img_jpeg.shape
		original_img_jpeg = original_img_jpeg[:h, :w-2]
		#rec_image_jpeg = cv2.imread(join(join(test_dir, reconstucted_JPEG_dir), f))
		print(compute_ssim(original_img_jpeg, rec_image_jpeg))
		del rec_image_jpeg

		# Original JPEG vs Reconstructed from PNG
		print("== Original PNG vs Reconstructed from PNG ==")
		#jpeg_name = f.split('.')[0] + ".jpeg"
		#original_img_jpeg = cv2.imread(join(join(test_dir, original_JPEG_dir), jpeg_name))
		#h, w, _ = original_img_jpeg.shape
		#original_img_jpeg = original_img_jpeg[:h, :w-2]
		rec_image_png = cv2.imread(join(join(test_dir, reconstucted_uncompressed_dir), f))
		print(compute_ssim(original_img_jpeg, rec_image_png))
		del rec_image_png

		# Original PNG vs Original JPEG
		print("== Original PNG vs Original JPEG ==")
		original_img_png = cv2.imread(join(source_filename_dir, f))
		jpeg_name = f.split('.')[0] + ".jpeg"
		original_img_jpeg = cv2.imread(join(join(test_dir, original_JPEG_dir), jpeg_name))
		print(compute_ssim(original_img_png, original_img_jpeg))
		del original_img_png, original_img_jpeg

		print("")
