import numpy as np
import math
import cv2
from tqdm import tqdm


PI = math.pi

RED = np.array([0,0,255])


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
def sinusoidal2latlng(x, y, img_w, img_h, lng0=0):
    w = img_w
    h = img_h
    xx = 2*math.pi*(x/w) - math.pi
    lat = math.pi*y/h - math.pi/2
    if y == 0:
        lng = 0
    else:
        lng = xx / math.cos(lat) + lng0
    return (lat, lng)


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
    lng = xx/math.cos(lat0) + lng0
    lat = yy + lat0
    return (lat, lng)


def rearrange_cartesian2latlng(x, y, img_w, img_h):
    row_length = img_w * math.cos( (PI/2) * ( 1-(y/(img_h/2)) ) )
    lng = 2*PI * ( (x-img_w//2 + row_length/2) / row_length ) - PI
    lat  = PI * y/(img_h//2) - PI/2
    return (lat, lng)



"""
Converts the latitude and logitude to sinusoidal coordinates.

In:
    lat: latitude in range [-PI/2, PI/2].
    longitude: longitude in range [-PI, PI].
    img_w: equirectangular projection image width.
    img_h: equirectangular projection image heigth.
    lng0: the meridian of the projection in radians.
Out:
    x, y: horizontal and vertical coordinates in the sinusoidal projection.
"""
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
            lat, lng = sinusoidal2latlng(x, y, img_w, img_h, lng0=lng0)
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











"""
Rearrange pixels in the sinusoidal projected image.

In:
    sin_img: the sinusoidal projected image.
Out:
    img_r: rearranged pixels from sin_img.
"""
def sinusoidal_rearrange_forward(sin_img):
    img = np.zeros_like(sin_img)
    img_h, img_w, _ = img.shape

    # Horizontal rearrangement
    for x in range(0,img_w):
        for y in range(0,img_h):
            d = abs((img_w/2) * math.cos(math.pi*(y/(img_h) - 0.5)))

            lat, lng = equirectangular2latlng(x, y, 2*d, img_h)
            xs, ys = latlng2sinusoidal(lat, lng, img_w, img_h)


            if y < img_h//2:
                img[y, x] = cv2.getRectSubPix(sin_img, (1,1), (x+(img_w//2-d),y))
            elif y == img_h//2 or y == img_h//2+1:
                img[y, x] = sin_img[y,x]
            else:
                img[y,x] =  cv2.getRectSubPix(sin_img, (1,1), (x-(img_w//2-d),y))


    # Vertical rearrangement
    pix_count = int(img_h*math.acos(0.5)/PI)
    img_r = np.zeros_like(img)
    for x in range(0,img_w):
        v_shift = abs(img_h*math.acos(x/img_w)/PI - img_h//2)
        for y in range(0,img_h):
            if y+v_shift+1 > 2*pix_count+2:
                continue
            img_r[y,x] = cv2.getRectSubPix(img, (1,1), (x,y+v_shift+1))#img[y,x]
    return img_r[:2*pix_count+2,:,:]



def sinusoidal_compression(equi_img):
    img = np.zeros_like(equi_img)
    img_h, img_w, _ = img.shape

    lng0 = 0
    pix_count = int(img_h*math.acos(0.5)/PI)
    # Sinusoidal projection and horizontal shift
    for x in range(0,img_w):
        for y in range(0,img_h):
            d = abs((img_w/2) * math.cos(math.pi*(y/img_h - 0.5)))
            if y <= img_h/2: # North hemisphere horizontal shift
                x_sin = x + (img_w/2 - d) # X coord in sinusoidal space
            else: # South hemisphere horizontal shift
                x_sin = x - (img_w/2 - d) # X coord in sinusoidal space

            lat, lng = sinusoidal2latlng(x_sin, y, img_w, img_h)
            x_equi, y_equi = latlng2equirectangular(lat, lng, img_w, img_h)

            img[y, x] = cv2.getRectSubPix(equi_img, (1,1), (x_equi, y_equi))

    # Vertical rearrangement
    pix_count = int(img_h*math.acos(0.5)/PI)
    img_r = np.zeros_like(img)
    for x in range(0,img_w):
        v_shift = abs(img_h*math.acos(x/img_w)/PI - img_h//2)
        for y in range(0,img_h):
            img_r[y,x] = cv2.getRectSubPix(img, (1,1), (x,y+v_shift+1))#img[y,x]

    return img_r[:2*pix_count+2,:,:]


def sinusoidal_decompression(img_r):
    h, w, _ = img_r.shape
    img = np.zeros((w//2, w, 3), dtype=np.uint8)
    img_h, img_w, _ = img.shape

    # Vertical rearrangement
    pix_count = int(img_h*math.acos(0.5)/PI)
    for x in range(0,img_w):
        v_shift = img_h*math.acos(x/img_w)/PI - img_h//2
        for y in range(0,img_h):
            img[y,x] = cv2.getRectSubPix(img_r, (1,1), (x,y+v_shift+1))#img[y,x]

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
