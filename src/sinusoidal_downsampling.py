import numpy as np
import math
import cv2


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
    lat, lng: latitude and longitude coordinates
"""
def sinusoidal2latlng(x, y, lng0, img_w, img_h):
    w = img_w
    h = img_h
    xx = 2*math.pi*(x/w) - math.pi
    lat = math.pi*y/h - math.pi/2
    lng = xx / math.cos(lat) + lng0
    return (lat, lng)



def equirectangular2latlng(x, y,  img_w, img_h, lat0=0, lng0=0):
    w = img_w
    h = img_h
    xx = 2*math.pi*x/w - math.pi
    yy = math.pi*y/h - math.pi/2
    lng = xx/math.cos(lat0) + lng0
    lat = yy + lat0
    return (lat, lng)



def latlng2sinusoidal(lat,lng, img_w, img_h, lng0=0):
    w = img_w
    h = img_h
    xx = (lng-lng0)*math.cos(lat)
    yy = lat
    x = w * (xx+math.pi) / (2*math.pi)
    y = h * (yy+math.pi/2) / math.pi
    return (x, y)





"""
Converts latitude and longitude coordinates to the equirectangular projection
coordinates

In:
    lat: latitude coordinate [-PI/2, PI/2].
    lng: longitude coordinate [-PI, PI].
    img_w: equirectangular projection image width.
    img_h: equirectangular projection image heigth.
    lat0: the equator of the projection in radians.
    lng0: the meridian of the projection in radians.
Out:
    x,y: the equirectangular image coordinates.
"""
def latlng2equirectangular(lat, lng, img_w, img_h, lat0=0, lng0=0):
    w = img_w
    h = img_h
    xx = (lng-lng0)*math.cos(lat0)
    yy = lat-lat0
    x = w * (xx+math.pi) / (2*math.pi)
    y = h * (yy+math.pi/2) / math.pi
    return (x, y)






"""
Downsample the image and generate a sinusoidal like shape frome the
equirectangular image.

In:
    img: the image size must be 2:1 proportion, grayscale, and a equirectangular projected image.
Out:
    img_r: a downsampled version of *img* converted into a sinusoidal projection.
"""
def sinusoidal_downsampling(equi_img):
    img = np.zeros_like(equi_img)
    img_h, img_w, _ = img.shape
    lng0 = 0

    for x in range(0,img_w):
        for y in range(0,img_h):
            d = abs((img_w/2) * math.cos(math.pi*(y/img_h - 0.5)))
            thresh_low = int(img_w/2 - d)
            thresh_high = int(img_w/2+d)
            if x < thresh_low or x > thresh_high: # Limit the sinusoidal shape
                continue
            lat, lng = sinusoidal2latlng(x, y, lng0, img_w, img_h)
            xs, ys = latlng2equirectangular(lat, lng, img_w, img_h)
            img[y, x] = cv2.getRectSubPix(equi_img, (1,1), (xs,ys))
    return img


def sinusoidal_reconstruction(sin_img):
    img = np.zeros_like(sin_img)
    img_h, img_w, _ = img.shape
    lng0 = 0

    for x in range(0,img_w):
        for y in range(0,img_h):
            lat, lng = equirectangular2latlng(x, y, img_w, img_h)
            xs, ys = latlng2sinusoidal(lat, lng, img_w, img_h)
            img[y, x] = cv2.getRectSubPix(sin_img, (1,1), (xs,ys))
    return img
