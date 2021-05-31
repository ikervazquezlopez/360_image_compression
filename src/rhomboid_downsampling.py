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
    h, w, _ = img.shape
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
    h, w, _ = img_r.shape
    I_erp = np.zeros_like(img_r)
    for i in range(w):
        for j in range(h):
            idxi = (i*_L(j,h)) / (2*h) - _L(j,h)/2 + h
            I_erp[j,i] = cv2.getRectSubPix(img_r, (1,1), (idxi,j))
    return I_erp




"""
Rearrange the triangle in the top right quadrant as stated in the paper but with
a slight modification for the index computation.

In:
    img: the downsampled rhomboid image.
    canvas: the image where the rearrangement is stored.
Out:
    canvas: the image where the rearrangement is stored.
"""
def rearrange_top_right_quadrant(img, canvas):
    h, w, _ = img.shape

    for j in range(h//2+1):
        for i in range(h, h+2*j):
            ip=i
            jp = j
            if i > h-1+j:
                ip = h-1+j
                jp = j-(i-ip)#j-i-ip
                #print(j, i, jp, ip)
            canvas[jp, ip-h//2] = img[j, i]
    return canvas




"""
Rearranges the full downsampled rhomboid shape as stated in the paper. It generates
a canvas and uses the top right rearrangement function to store the rearranged image.
Since the rearrange_top_right_quadrant() only rearranges the top right quadrant,
this method flips the img_rh and canvas accordingly to generate the correct rearrangement.

In:
    img_rh: the downsampled rhomboid image.
Out:
    canvas: the rearranged downsampled rhomboid image of size (h,h)
"""
def rhomboid_rearrangement(img_rh):
    h, w, _ = img_rh.shape

    canvas = np.zeros((h, h, 3), np.uint8)

    img = img_rh
    canvas = rearrange_top_right_quadrant(img, canvas)

    img = cv2.flip(img,0)
    canvas = cv2.flip(canvas,0)
    canvas = rearrange_top_right_quadrant(img, canvas)

    img = cv2.flip(img,1)
    canvas = cv2.flip(canvas,1)
    canvas = rearrange_top_right_quadrant(img, canvas)

    img = cv2.flip(img,0)
    canvas = cv2.flip(canvas,0)
    canvas = rearrange_top_right_quadrant(img, canvas)

    canvas = cv2.flip(canvas,1)
    return canvas


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
    h, w, _ = img.shape
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
    h, w, _ = img_r.shape
    I_erp = np.zeros_like(img_r)
    for i in range(w):
        for j in range(h):
            idxi = (i*_L_1_1(j,h)) / (h) - _L_1_1(j,h)/2 + h//2
            I_erp[j,i] = cv2.getRectSubPix(img_r, (1,1), (idxi,j))
    return I_erp
