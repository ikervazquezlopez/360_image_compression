import numpy as np
import math
import cv2


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
def sinusoidal2latlng(x, y, lng0, img_w, img_h):
    w = img_w
    h = img_h
    xx = 2*math.pi*(x/w) - math.pi
    lat = math.pi*y/h - math.pi/2
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

            if x+(img_w-2*int(d)) > img_w:
                continue

            lat, lng = equirectangular2latlng(x, y, 2*d, img_h)
            xs, ys = latlng2sinusoidal(lat, lng, img_w, img_h)

            if y < img_h//2:
                img[y, x] = cv2.getRectSubPix(sin_img, (1,1), (xs,y))
            elif y == img_h//2:
                img[y, x] = sin_img[y,x]
            else:
                img[y, x+(img_w-2*int(d))-2] =  cv2.getRectSubPix(sin_img, (1,1), (xs,y))


    # Vertical rearrangement
    pix_count = int(img_h*math.acos(0.5)/PI)
    img_r = np.zeros_like(img)
    for x in range(0,img_w):
        for y in range(0,img_h):
            v_shift = int(img_h*math.acos(x/img_w)/PI - img_h//2)
            img_r[y+v_shift-1,x] = cv2.getRectSubPix(img, (1,1), (x,y))#img[y,x]

    return img_r[:2*pix_count+2,:,:]



def sinusoidal_rearrange_backward(r_img):
    img = np.zeros_like(r_img)
    img_h, img_w, _ = img.shape
    for x in range(0,img_w):
        for y in range(0,img_h):
            d = abs((img_w/2) * math.cos(math.pi*(y/img_h - 0.5)))

            if x+(img_w//2-int(d))-1 < 0 or x+(img_w//2-int(d))-1 >= img_w: # Limit the sinusoidal shape
                continue

            lat, lng = equirectangular2latlng(x, y, img_w, img_h)
            xs, ys = latlng2sinusoidal(lat, lng, img_w, img_h)
            if y < img_h//2 :
                img[y, x+(img_w//2-int(d))-1] = r_img[y, x]
            else:
                img[y, (img_w//2-int(d))-x] = r_img[y, img_w-x-1]
    return img
