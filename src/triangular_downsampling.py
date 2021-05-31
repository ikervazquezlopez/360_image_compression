import cv2
import numpy as np
import sys

SAMPLING_INCREMENT_2_1 = 4
SAMPLING_INCREMENT_1_1 = SAMPLING_INCREMENT_2_1 // 2

"""
pano = cv2.imread(sys.argv[1])
h, w, c = pano.shape

# Create output image which half height and width+1 to fit both triangles
canvas = np.zeros( (int(h/2), w-2, c), dtype=np.uint8)

# Compress and fill the northern hemisphere
for r in range(0, int(h/2)):
    chunks = 1 + r * d
    chunk_size =  w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        canvas[r,c] = pano[r, c_start]

# Flip horizontally and vertically
pano = cv2.flip(pano, -1)
canvas = cv2.flip(canvas, -1)

# Compress and fill the southern hemisphere after flipping
for r in range(0, int(h/2)):
    chunks = 1 + r * d
    chunk_size =  w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        #canvas[int(h/2)-r-1,w-c-3] = pano[h-r-1, w-c_start-1]
        canvas[r,c] = pano[r, c_start]

# Flip canvas back
canvas = cv2.flip(canvas, -1)

cv2.imwrite(sys.argv[2], canvas)
"""

def row_linear_interpolation(p, r, c, c_start, chunk_size, img):
    d_start = (p - c_start) / chunk_size
    d_end = 1 - d_start
    color = d_start*img[r, c] + d_end*img[r, c]
    #color = cv2.getRectSubPix(img, (1,1), (p,r))
    return color.astype(np.uint8)



def triangular_downsampling_2_1(img):
    h, w, c = img.shape

    # Create output image which half height and width+1 to fit both triangles
    canvas = np.zeros( (int(h/2), w-2, c), dtype=np.uint8)

    # Compress and fill the northern hemisphere
    for r in range(0, int(h/2)):
        chunks = 1 + r * SAMPLING_INCREMENT_2_1
        chunk_size =  w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            canvas[r,c] = img[r, c_start]

    # Flip horizontally and vertically
    img = cv2.flip(img, -1)
    canvas = cv2.flip(canvas, -1)

    # Compress and fill the southern hemisphere after flipping
    for r in range(0, int(h/2)):
        chunks = 1 + r * SAMPLING_INCREMENT_2_1
        chunk_size =  w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            #canvas[int(h/2)-r-1,w-c-3] = pano[h-r-1, w-c_start-1]
            canvas[r,c] = img[r, c_start]

    # Flip canvas back
    canvas = cv2.flip(canvas, -1)
    return canvas


def triangular_downsampling_1_1(img):
    h, w, c = img.shape
    print(img.shape)

    # Create output image which half height and width+1 to fit both triangles
    canvas = np.zeros( (int(h/2), w, c), dtype=np.uint8)

    # Compress and fill the northern hemisphere
    for r in range(0, int(h/2)):
        chunks = 1 + r * SAMPLING_INCREMENT_1_1
        chunk_size =  w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            canvas[r,c] = img[r, c_start]

    # Flip horizontally and vertically
    img = cv2.flip(img, -1)
    canvas = cv2.flip(canvas, -1)

    # Compress and fill the southern hemisphere after flipping
    for r in range(0, int(h/2)):
        chunks = 1 + r * SAMPLING_INCREMENT_1_1
        chunk_size =  w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            #canvas[int(h/2)-r-1,w-c-3] = pano[h-r-1, w-c_start-1]
            canvas[r,c] = img[r, c_start]

    # Flip canvas back
    canvas = cv2.flip(canvas, -1)
    return canvas




def triangular_reconstruction_2_1(img):
    h, w, c = img.shape

    canvas = np.zeros( (h*2, w, c), dtype=np.uint8)
    print(canvas.shape)

    # Build middle rows without any interpolation
    for c in range(0, w-2):
        canvas[h-1, c] = img[h-1, c]
        canvas[h, c] = img[0, c+2]

    # Reconstruct northern hemisphere
    for r in range(h-1, -1, -1):
        chunks = 1 + r * SAMPLING_INCREMENT_2_1
        chunk_size = w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            c_end = c_start + int(chunk_size)
            canvas[r, c_start] = img[r, c]
            for p in range(c_start+1, c_end+1):
                if p >= w:
                    continue
                canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size), img)

    # Flip horizontally and vertically
    img = cv2.flip(img, -1)
    canvas = cv2.flip(canvas, -1)

    # Reconstruct the southern hemisphere
    for r in range(h-1, -1, -1):
        chunks = 1 + r * SAMPLING_INCREMENT_2_1
        chunk_size = w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            c_end = c_start + int(chunk_size)
            canvas[r, c_start] = img[r, c]
            for p in range(c_start+1, c_end+1):
                if p >= w:
                    continue
                canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size), img)

    # Flip canvas back
    canvas = cv2.flip(canvas, -1)
    return canvas


def triangular_reconstruction_1_1(img):
    h, w, c = img.shape

    canvas = np.zeros( (h*2, w, c), dtype=np.uint8)

    # Build middle rows without any interpolation
    for c in range(0, w-2):
        canvas[h-1, c] = img[h-1, c]
        canvas[h, c] = img[0, c+2]

    # Reconstruct northern hemisphere
    for r in range(h-1, -1, -1):
        chunks = 1 + r * SAMPLING_INCREMENT_1_1
        chunk_size = w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            c_end = c_start + int(chunk_size)
            canvas[r, c_start] = img[r, c]
            for p in range(c_start+1, c_end+1):
                if p >= w:
                    continue
                canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size), img)

    # Flip horizontally and vertically
    img = cv2.flip(img, -1)
    canvas = cv2.flip(canvas, -1)

    # Reconstruct the southern hemisphere
    for r in range(h-1, -1, -1):
        chunks = 1 + r * SAMPLING_INCREMENT_1_1
        chunk_size = w / chunks
        for c in range(0, chunks):
            c_start = int(c*chunk_size)
            c_end = c_start + int(chunk_size)
            canvas[r, c_start] = img[r, c]
            for p in range(c_start+1, c_end+1):
                if p >= w:
                    continue
                canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size), img)

    # Flip canvas back
    canvas = cv2.flip(canvas, -1)
    return canvas
