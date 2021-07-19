import numpy as np
NUMBA_ENABLE_CUDASIM = 1
from numba import jit, vectorize, float64, float32, int32, uint8, cuda
from pdb import set_trace
import math
import cv2
from tqdm import tqdm

import operator


PI = math.pi



"""
Converts the sinusoidal coordinates to latitude and longitude coordinates.

In:
    x: sinusoidal image horizontal coordinate.
    y: sinusoidal image vertical coodinate
    img_w: equirectangular projection image width.
    img_h: equirectangular projection image heigth.
    lng0: the meridian of the projection in radians.
Out:
    d_out: latitude and longitude coordinates (1D array length 2)
"""
@cuda.jit(device=True)
def sinusoidal2latlng(x, y, img_w, img_h, d_out):
    lng0 = 0
    w = img_w
    h = img_h
    xx = 2*math.pi*(x/w) - math.pi
    lat = math.pi*y/h - math.pi/2
    if y == 0:
        lng = 0
    else:
        lng = xx / math.cos(lat) + lng0
    d_out[0] = lat
    d_out[1] = lng


"""
Converts the equirectanuglar coordinates to longitude and latitude.

In:
    x: equirectangular image horizontal coordinate.
    y: equirectangular image vertical coodinate
    img_w: equirectangular projection image width.
    img_h: equirectangular projection image heigth.
    lng0: the meridian of the projection in radians.
Out:
    lat, lng: latitude and longitude coordinates
"""
def equirectangular2latlng(x, y,  img_w, img_h, lat0=0, lng0=0):
    w = img_w
    h = img_h
    xx = 2*math.pi*x/w - math.pi
    yy = math.pi*y/h - math.pi/2
    lng = xx/np.cos(lat0) + lng0
    lat = yy + lat0
    return (lat, lng)




"""
Converts the latitude and logitude to sinusoidal coordinates.

In:
    lat: latitude in range [-PI/2, PI/2].
    longitude: longitude in range [-PI, PI].
    img_w: equirectangular projection image width.
    img_h: equirectangular projection image heigth.
Out:
    d_out: sinusoidal coordinates (1D array length 2)
"""
@cuda.jit(device=True)
def latlng2sinusoidal(lat,lng, img_w, img_h, d_out):
    lng0 = 0
    w = img_w
    h = img_h
    xx = (lng-lng0)*math.cos(lat)
    yy = lat
    d_out[0] = w * (xx+math.pi) / (2*math.pi)
    d_out[1] = h * (yy+math.pi/2) / math.pi





"""
Converts latitude and longitude coordinates to the equirectangular projection
coordinates

In:
    lat: latitude coordinate [-PI/2, PI/2].
    lng: longitude coordinate [-PI, PI].
    img_w: equirectangular projection image width.
    img_h: equirectangular projection image heigth.
Out:
    d_out: equirectangular coordinates (1D array length 2)
"""
@cuda.jit(device=True)
def latlng2equirectangular(lat, lng, img_w, img_h, d_out):
    lat0 = 0
    lng0 = 0
    w = img_w
    h = img_h
    xx = (lng-lng0)*math.cos(lat0)
    yy = lat-lat0
    x = w * (xx+math.pi) / (2*math.pi)
    y = h * (yy+math.pi/2) / math.pi
    d_out[0] = x
    d_out[1] = y




"""
Assigns the *src* pixel values to *dst* pixels.

In:
    src: input array (lenth 3) pixel.
Out:
    dst: output array (length 3) pixel.
"""
@cuda.jit(device=True)
def assign_pixel(src, dst):
    dst[0] = src[0]
    dst[1] = src[1]
    dst[2] = src[2]



"""
Linear interpolation between *c0* and *c1*.

In:
    c0: 3 channel array, reference color pixel.
    c1: 3 channel array, reference color pixel.
    d: distance to to the point (from c0 coordinates) that it is going to be interpolated
Out:
    out: 3 channel array where the computed color will be stored.
"""
#@cuda.jit((uint8[:], uint8[:], float32, uint8[:]), device=True)
@cuda.jit(device=True)
def interpolate(c0, c1, d, out):
    out[0] = math.floor(c0[0] * (1-d) + c1[0] * d)
    out[1] = math.floor(c0[1] * (1-d) + c1[1] * d)
    out[2] = math.floor(c0[2] * (1-d) + c1[2] * d)



"""
Returns, in *color* (3 channel array), the sub pixel color value at (x,y) float
coordinates from the *img* image.

In:
    x: x (float) coordinate.
    y: y (float) coordinate.
    img: input image to get the sub pixel color from.
    img_w: image width.
    img_h: image height.
Out:
    color: 3 channel array where the computed color will be stored.
"""
@cuda.jit(device=True)
def getSubPixel(x, y, img, w, h, color):
    img_w = w
    img_h = h
    x0 = math.floor(x)
    y0 = math.floor(y)

    top_color = cuda.local.array((3), uint8)
    bot_color = cuda.local.array((3), uint8)

    dx = x-x0
    dy = y-y0

    x0 = int(x0)
    y0 = int(y0)

    xp = int(x)
    yp = int(y)

    # Corner cases
    if (y0-1 < 0 and x0-1 < 0) or (y0-1 < 0 and x0+1 >= img_w):
        assign_pixel(img[yp,xp], top_color)
        assign_pixel(img[yp+1, xp], bot_color)
        #top_color = img[y,x]
        #bot_color = img[y+1, x]
    elif (y0+1 >= img_h and x0+1 >= img_w) or (y0+1 >= img_h and x0-1 < 0):
        assign_pixel(img[yp-1,xp], top_color)
        assign_pixel(img[yp, xp], bot_color)
        #top_color = img[y-1,x]
        #bot_color = img[y, x]



    # Side cases
    elif y0-1 < 0:
        interpolate(img[yp, x0-1], img[yp, x0+1], dx, top_color)
        interpolate(img[yp+1, x0-1], img[yp+1, x0+1], dx, bot_color)
    elif y0+1 >= img_h:
        interpolate(img[yp-1, x0-1], img[yp-1, x0+1], dx, top_color)
        interpolate(img[yp, x0-1], img[yp, x0+1], dx, bot_color)
    elif x0+1 >= img_w:
        interpolate(img[yp-1, x0-1], img[yp-1, x0], dx, top_color)
        interpolate(img[yp+1, x0-1], img[yp+1, x0], dx, bot_color)
    elif x0-1 < 0:
        interpolate(img[yp-1, x0], img[yp-1, x0+1], dx, top_color)
        interpolate(img[yp+1, x0], img[yp+1, x0+1], dx, bot_color)

    # General case
    else:
        interpolate(img[yp-1, x0-1], img[yp-1, x0+1], dx, top_color)
        interpolate(img[yp+1, x0-1], img[yp+1, x0+1], dx, bot_color)

    interpolate(top_color, bot_color, dy, color)




"""
Projects the input equirectangular image to a sinusoidal projected image and
horizontally rearranges the pixels to fill the empty spaces. Right after,
*sinusoidal_compression_1* must be called to produce the compressed version
of the input image.

In:
    d_equi_img: the input equirectangular image (device pointer)
    d_img_r: the output compressed image (device_pointer)
    d_tmp: the temporal image where the horizontal rearrangement is computed (device pointer)
    d_dim: input image dimensions (width, height) (device pointer)
"""
@cuda.jit
def sinusoidal_compression_0(d_equi_img, d_img_r, d_tmp, d_dim):
    img_w = d_dim[0]
    img_h = d_dim[1]
    img_w_half = img_w/2
    img_h_half = img_h/2
    img_w_half_int = int(img_w_half)

    lng0 = 0
    coords_sin = cuda.local.array((2), float32)
    coords_equi = cuda.local.array((2), float32)

    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    # Sinusoidal projection and horizontal shift
    while ty < img_h:
        while tx < img_w:
            d = abs(img_w_half * math.cos(math.pi*(ty/img_h - 0.5)))
            if ty <= img_h_half: # North hemisphere horizontal shift
                x_sin = tx + (img_w_half - d) # X coord in sinusoidal space
            else: # South hemisphere horizontal shift
                x_sin = tx - (img_w_half - d) # X coord in sinusoidal space

            sinusoidal2latlng(x_sin, ty, img_w, img_h, coords_sin)
            latlng2equirectangular(coords_sin[0], coords_sin[1], img_w, img_h, coords_equi)

            getSubPixel(coords_equi[0], coords_equi[1], d_equi_img, img_w, img_h, d_tmp[ty,tx])

            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.gridDim.x



"""
Vertically shifts the pixels to generate the compressed version of the image.
Must be preceeded by *sinusoidal_compression_0" kernel.

In:
    d_equi_img: the input equirectangular image (device pointer)
    d_img_r: the output compressed image (device_pointer)
    d_tmp: the temporal image where the horizontal rearrangement is computed (device pointer)
    d_dim: input image dimensions (width, height) (device pointer)
"""
@cuda.jit
def sinusoidal_compression_1(d_equi_img, d_img_r, d_tmp, d_dim):
    img_w = d_dim[0]
    img_h = d_dim[1]
    img_w_half = img_w/2
    img_h_half = img_h/2
    img_w_half_int = int(img_w_half)

    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x

    f_img_h = float(img_h)

    f_img_w = float(img_w)
    f_img_h_half = float(img_h_half)

    # Vertical shift
    while tx < img_w:
        f_tx = float(tx)
        v_shift = abs(f_img_h*math.acos(f_tx/f_img_w)/math.pi - f_img_h_half)

        while ty < img_h:
            typ = float(ty)+v_shift+1.0

            if typ < img_h:
                getSubPixel(tx, typ, d_tmp, img_w, img_h, d_img_r[ty,tx])

            ty = ty + cuda.gridDim.x
        tx = tx + cuda.blockDim.x
        ty = cuda.blockIdx.x





"""
Decompressed the image produced by 'sinusoidal_compression' function into an
equirectangular image

In:
    img_r: the compressed sinusoidal image.
Out:
    equi: the decompressed equirectangular image.
"""
def sinusoidal_decompression(img_r):
    h, w, _ = img_r.shape
    img = np.zeros((w//2, w, 3), dtype=np.uint8)
    img_h, img_w, _ = img.shape

    # Vertical rearrangement
    pix_count = int(img_h*math.acos(0.5)/PI)
    for x in range(0,img_w):
        v_shift = img_h*math.acos(x/img_w)/PI - img_h//2
        for y in range(0,img_h):
            img[y,x] = cv2.getRectSubPix(img_r, (1,1), (x,y+v_shift+1))

    # Horizontal rearrangement
    out = np.zeros_like(img)
    for x in range(0,img_w):
        for y in range(0,img_h//2+1):
            d = abs((img_w/2) * math.cos(math.pi*(y/img_h - 0.5)))

            if x < img_w//2-d or x > img_w//2+d: # Limit the sinusoidal shape
                continue

            if y >= img_h//2-1 :
                out[y,x] = img[y,x]
            else:
                out[y, x] = cv2.getRectSubPix(img, (1, 1), (x-(img_w//2-d),y))


    out = cv2.flip(out, -1)
    img = cv2.flip(img, -1)
    for x in range(0,img_w):
        for y in range(0,img_h//2+1):
            d = abs((img_w/2) * math.cos(math.pi*(y/img_h - 0.5)))

            if x < img_w//2-d or x > img_w//2+d: # Limit the sinusoidal shape
                continue

            if y >= img_h//2-1 :
                out[y,x] = img[y,x]
            else:
                out[y, x] = cv2.getRectSubPix(img, (1, 1), (x-(img_w//2-d),y))

    out = cv2.flip(out, -1)

    equi = np.zeros_like(out)
    for x in range(0, img_w):
        for y in range(0, img_h):
            lat, lng = equirectangular2latlng(x, y, img_w, img_h)
            xs, ys = latlng2sinusoidal(lat, lng, img_w, img_h)
            equi[y,x] = cv2.getRectSubPix(out, (1, 1), (xs,ys))
    return equi
