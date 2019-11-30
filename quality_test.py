import cv2
import numpy as np
import math

def show_image(img):
	cv2.imshow("test", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def MSE(img0, img1):
	#img = cv2.subtract(img0, img1)
	img = np.subtract(img0, img1)
	img = img.astype(dtype=np.float32)#/255
	img = img*img
	print(np.max(img))
	MSE = np.sum(img) / img.shape[0] * img.shape[1] * img.shape[2]
	return MSE

def PSNR(img0, img1):
	i0 = cv2.cvtColor(img0, cv2.COLOR_RGB2YCR_CB)
	i1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
	mse = MSE(i0, i1)
	max = np.max(i0)
	psnr = 20*math.log10(max) - 10*math.log10(mse)
	psnr = 10*math.log10(max*max/mse)
	psnr = 20*math.log10(max/math.sqrt(mse))
	return psnr



original = cv2.imread('data/test.jpeg')
print(original.shape)
comp = cv2.imread('reconstructed.png')
#comp = cv2.imread('data/test.jpeg')
h, w, c = comp.shape
comp = comp[:h, :w-1]
#comp = comp[:h, :w]


#print(PSNR(original, comp))
print(cv2.PSNR(original, comp))



#diff = cv2.subtract(original, comp)
#diff = diff * 2
#cv2.imshow("test", diff)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
