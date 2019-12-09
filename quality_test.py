import cv2
import numpy as np
import math
import skimage.measure as metrics
import cupy as cp




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
	del original
	img0_Y = cp.array(img0_Y)
	print("Loading image 2")
	img1_Y = cv2.cvtColor(comp, cv2.COLOR_BGR2YCR_CB)[:,:,0]
	del comp
	img0_Y = cp.array(img0_Y)

	print("Computing means")
	mux = cp.mean(img0_Y)
	muy = cp.mean(img1_Y)
	print("Computing variances")
	sigma2x = cp.var(img0_Y)
	sigma2y = cp.var(img1_Y)
	print("Computing covariance")
	sigmaxy = cp.cov(img0_Y, img1_Y)

	# Compute SSMI
	print("Computing SSIM")
	numerator = (2*mux*muy + c1) * (2*sigmaxy + c2)
	denominator = (mux*mux + muy*muy + c1) * (sigma2x*sigma2x + sigma2y*sigma2y + c2)
	ssim = numerator / denominator
	return ssim



original = cv2.imread('testing/uncompressed_original/original.png')
h, w, c = original.shape
original = original[:h, :w-2]
print(original.shape)
comp = cv2.imread('testing/uncompressed_reconstructed/reconstructed.png')
print(comp.shape)
#comp = cv2.imread('data/test.jpeg')
h, w, c = comp.shape
#comp = comp[:h, :w]
#comp = comp[:h, :w]


#print(PSNR(original, comp))
print(cv2.PSNR(original, comp))
print(SSIM(original,comp))



#diff = cv2.subtract(original, comp)
#diff = diff * 2
#cv2.imshow("test", diff)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
