import numpy as np
import cv2





"""
=======================================================================================================================
Lee2017 et al,
Omnidirectional video coding using latitude adaptive down-sampling and pixel rearrangement
"""


"""
Helper function used for the downsampling.
In:
 int i: pixel row.
 int j: pixel col.
 int N: height in pixels of the image.
Out:
    float:
"""
def _u(i, j, N):
    return ((N + _L(j,N) - 1) / 2) - i


"""
Length of the j-th row.

In:
    int j: the image row index.
    int N: height in pixels of the image.
Out:
    int: j-th row length
"""
def _L(j, N):
    return 2*(N - abs(2*j-N+1))



"""
Sampling rate for j-th row.

In:
    int j: the image row index to be downsampled
    int N: height in pixels of the image
Out:
    float: sampling rate.
"""
def _D(j, N):
    return (2*N) / _L(j, N)

"""
Downsample the image and generate a rhomboid like shape frome the
equirectangular image.

In:
    img: the image size must be 2:1 proportion, grayscale, and a equirectangular projected image.
Out:
    img_r: a downsampled version of *img* converted into a rhomboid.
"""
def rhomboid_downsample(img):
    h, w = img.shape
    I_r = np.zeros_like(img)
    for i in range(w):
        for j in range(h):
            idxi = _D(j,h) * _u(i-h//2,j,h)
            I_r[j,i] = cv2.getRectSubPix(img, (1,1), (idxi,j))
    return I_r


"""
Reconstruct the equirectangular projected image from the downsampled rhomboid image.

In:
    img_r: downsampled rhomboid image.
Out:
    img: reconstructed equirectangular image.
"""
def rhomboid_reconstruction(img_r):
    h, w = img_r.shape
    I_erp = np.zeros_like(img_r)
    for i in range(w):
        for j in range(h):
            idxi = (i*_L(j,h)) / (2*h) - _L(j,h)/2 + h
            I_erp[j,i] = cv2.getRectSubPix(img_r, (1,1), (idxi,j))
    return I_erp

"""
=======================================================================================================================
"""



"""
Sampling rate for j-th row for images of 1:1 rate.

In:
    int j: the image row index to be downsampled
    int N: height in pixels of the image
Out:
    float: sampling rate.
"""
def _L_1_1(j, N):
    return (N - abs(2*j-N+1))#N - abs(2*j-N+1)



"""
Length of the j-th row for images of 1:1 rate.

In:
    int j: the image row index.
    int N: height in pixels of the image.
Out:
    int: j-th row length
"""
def _D_1_1(j, N):
    return (2*N) / _L(j, N)



"""
Helper function used for the downsampling for images of 1:1 rate.
In:
    int i: pixel row.
    int j: pixel col.
    int N: height in pixels of the image.
Out:
    float:
"""
def _u_1_1(i, j, N):
    return ((N//2 + _L_1_1(j,N) - 1) / 2 ) - i



"""
Downsample the image and generate a rhomboid like shape frome the
equirectangular image half image for images of 1:1 rate.

In:
    img: the image size must be 1:1 proportion, grayscale, and a half of an equirectangular projected image.
Out:
    img_r: a downsampled version of *img* converted into a rhomboid.
"""
def rhomboid_downsample_1_1(img):
    h, w = img.shape
    I_r = np.zeros_like(img)
    for i in range(w):
        for j in range(h):
            idxi = _D_1_1(j,h) * _u_1_1(i-h//4,j,h)
            I_r[j,i] = cv2.getRectSubPix(img, (1,1), (idxi,j))
    return I_r



"""
Reconstruct the equirectangular projected image from the downsampled rhomboid image for images of 1:1 rate.

In:
    img_r: downsampled rhomboid image.
Out:
    img: reconstructed half equirectangular image.
"""
def rhomboid_reconstruction_1_1(img_r):
    h, w = img_r.shape
    I_erp = np.zeros_like(img_r)
    for i in range(w):
        for j in range(h):
            idxi = (i*_L_1_1(j,h)) / (h) - _L_1_1(j,h)/2 + h//2
            I_erp[j,i] = cv2.getRectSubPix(img_r, (1,1), (idxi,j))
    return I_erp
