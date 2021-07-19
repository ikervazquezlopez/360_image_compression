import cv2
import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

NUMBA_ENABLE_CUDASIM = 1

import numba
from numba import jit, vectorize, float64, int32, cuda

import equirectangular_downsampling as ed
import rhomboid_downsampling as rd
import triangular_downsampling as td
import sinusoidal_downsampling as sd

import sinusoidal_downsampling_gpu as sin_gpu

import time


pano = cv2.imread("../out/PNG/equirectangular/Bluebonnet-0-L.png")
pano = cv2.resize(pano, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

sin_pano = cv2.imread("../out/PNG/sinusoidal/Bluebonnet-0-L.png")
sin_pano = cv2.resize(sin_pano, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)
print(pano.shape)



"""
================================================================================
Test split
"""

"""
left, right = ed.split(pano)
print(left.shape, right.shape)

cv2.imshow("left", left)
cv2.imshow("right", right)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
================================================================================
"""




"""
================================================================================
Test Lee Downsampling
"""

"""
I_r = rd.rhomboid_downsample(pano)
I_erp = rd.rhomboid_reconstruction(I_r)
I_c = rd.rhomboid_rearrangement(I_r)
cv2.imshow("Rhomboid rearrangement", I_c)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("out.png", I_c)
"""


"""
================================================================================
"""




"""
================================================================================
Test Rhomboid Downsampling (1,1)
"""

"""
left, right = ed.split(pano)
I_r = rd.rhomboid_downsample_1_1(left)


tmp = np.zeros_like(left)
tmp = rd.rhomboid_sort_northen_hemisphere_1_1(I_r,tmp)
I_r = cv2.flip(I_r, 0)
tmp = cv2.flip(tmp, 0)
I_s = rd.rhomboid_sort_northen_hemisphere_1_1(I_r,tmp)
I_s = cv2.flip(I_s, 0)
I_r = cv2.flip(I_r, 0)
"""
"""
I_c = rd.rhomboid_rearrangement_1_1(I_r)


I_erp = rd.rhomboid_reconstruction_1_1(I_r)
I_erp = cv2.flip(I_erp, 1)
cv2.imshow("Rhomboid rearrangement", I_c)
cv2.imshow("left", I_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("out.png", I_c)
"""

"""
================================================================================
"""




"""
================================================================================
Test Triangular Downsampling (2,1)
"""

"""
I_t = td.triangular_downsampling_2_1(pano)
I_erp = td.triangular_reconstruction_2_1(I_t)
cv2.imshow("Triangular reconstruction", I_erp)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
================================================================================
"""





"""
================================================================================
Test Triangular Downsampling (1,1)
"""

"""
left, right = ed.split(pano)
I_t = td.triangular_downsampling_1_1(left)
I_erp = td.triangular_reconstruction_1_1(I_t)
cv2.imshow("Triangular downsample", I_erp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("left.png", I_erp)
"""


"""
================================================================================
"""



"""
================================================================================
Test Sinusoidal Downsampling (2,1)
"""

"""
I_s = sd.sinusoidal_downsampling(pano)
#I_r = sd.sinusoidal_reconstruction(I_s)
print(pano.shape)
print(I_s.shape)
cv2.imshow("Pano", pano)
cv2.imshow("Sinusoidal downsample", I_s)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
================================================================================
"""








"""
================================================================================
Test Sinusoidal Compression-Decompression (2,1)
"""

"""
I_c = sd.sinusoidal_compression(pano)
I_rb = sd.sinusoidal_decompression(I_c)
cv2.imshow("Sinusoidal compressioon", I_c)
cv2.imshow("Sinusoidal rearrange backward", I_rb)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("rearrangement_backward.png", I_rb)
"""



"""
================================================================================
"""
















"""
================================================================================
Test Cuda interpolate device function
"""

"""
@cuda.jit
def foo_kernel(d_c0, d_c1, d_d, d_color):
    d_color = sin_gpu.interpolate(d_c0, d_c1, d_d, d_color)



stream = cuda.stream()

c0 = np.array([10,20,3], dtype=np.uint8)
c1 = np.array([5,10,0], dtype=np.uint8)
d = np.array([0.7], dtype=np.float32)
color = np.zeros_like(c0)

d_c0 = cuda.to_device(np.ascontiguousarray(c0), stream=stream)
d_c1 = cuda.to_device(np.ascontiguousarray(c1), stream=stream)
d_d = cuda.to_device(np.ascontiguousarray(d), stream=stream)
d_color = cuda.to_device(np.ascontiguousarray(color), stream=stream)

foo_kernel[32, 32](d_c0, d_c1, d_d, d_color)

color = d_color.copy_to_host()
print(color)
"""

"""
================================================================================
"""



"""
================================================================================
Test Cuda getSubPixel device function
"""

"""
@cuda.jit
def foo_kernel(d_x, d_y, d_img, d_w, d_h, d_color):
    sin_gpu.getSubPixel(d_x, d_y, d_img, d_w, d_h, d_color)



stream = cuda.stream()

h, w, _ = pano.shape
c1 = np.array([5,10,0], dtype=np.uint8)
x = np.array([466], dtype=np.uint32)
y = np.array([47], dtype=np.uint32)
color = np.array([0,0,0], dtype=np.float32)

d_pano = cuda.to_device(np.ascontiguousarray(pano), stream=stream)
d_w = cuda.to_device(np.ascontiguousarray(w), stream=stream)
d_h = cuda.to_device(np.ascontiguousarray(h), stream=stream)
d_x = cuda.to_device(np.ascontiguousarray(x), stream=stream)
d_y = cuda.to_device(np.ascontiguousarray(y), stream=stream)
d_color = cuda.to_device(np.ascontiguousarray(color), stream=stream)

foo_kernel[2, 2](d_x, d_y, d_pano, d_w, d_h, d_color)

color = d_color.copy_to_host()
print(color)
"""

"""
================================================================================
"""


"""
================================================================================
Test Cuda sinusoidal_compression kernel
"""

t0 = time.time()
stream = cuda.stream()

h, w, _ = pano.shape
dim = np.array([w,h], dtype=np.uint32)
img_r = np.zeros_like(pano)
tmp = np.zeros_like(pano)

out = np.zeros((h,w,4), dtype=np.int32)


d_pano = cuda.to_device(np.ascontiguousarray(pano), stream=stream)
d_img_r = cuda.to_device(np.ascontiguousarray(img_r), stream=stream)
d_tmp = cuda.to_device(np.ascontiguousarray(tmp), stream=stream)
d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)

d_out = cuda.to_device(np.ascontiguousarray(out), stream=stream)


sin_gpu.sinusoidal_compression_0[512, 512](d_pano, d_img_r, d_tmp, d_dim)
sin_gpu.sinusoidal_compression_1[512, 512](d_pano, d_img_r, d_tmp, d_dim)
img_r = d_img_r.copy_to_host()
pix_count = int(h*math.acos(0.5)/math.pi)
img_r = img_r[:2*pix_count+2,:,:]
tmp = d_tmp.copy_to_host()

out = d_out.copy_to_host()

print(out[0,512])
#plt.plot(out)
#plt.imshow(out)
#plt.colorbar()
#plt.show()

t = time.time()-t0
#cv2.imshow("Rearranged GPU", tmp)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('out.png', img_r)
print("{} seconds".format(t))


"""
================================================================================
"""
