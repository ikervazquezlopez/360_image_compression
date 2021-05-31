import numpy as np
import cv2



"""
Vertically splits the input image 'img' in two images.
It returns a tuple of two images, the left and right halves of the input image.

In:
    img: the image size must be 2:1 proportion, grayscale, and a equirectangular projected image.
Out:
    (img0, img1): A tuple of two images, the left and right halves of *img*
"""
def split(img):
    h, w, _ = img.shape
    center = w // 2
    left = img[:, 0:center]
    right = img[:, center:]
    return (left, right)



def downsample(img):
    return

def rearrange(img):
    return

def join_triangles(img0, img1):
    return

def process_image(img):
    return
