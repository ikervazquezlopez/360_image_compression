import cv2
import numpy as np
import sys

d = 4 # Increment rate per row


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
