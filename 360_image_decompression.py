import cv2
import numpy as np
import sys

d = 4 # Increment rate per row

def row_linear_interpolation(p, r, c, c_start, chunk_size):
    d_start = (p - c_start) / chunk_size
    d_end = 1 - d_start
    color = d_start*cpano[r, c] + d_end*cpano[r, c]
    return color.astype(np.uint8)


cpano = cv2.imread(sys.argv[1])
h, w, c = cpano.shape

canvas = np.zeros( (h*2, w, c), dtype=np.uint8)

# Build middle rows without any interpolation
for c in range(0, w-2):
    canvas[h-1, c] = cpano[h-1, c]
    canvas[h, c] = cpano[0, c+2]

# Reconstruct northern hemisphere
for r in range(h-1, 0, -1):
    chunks = 1 + r * d
    chunk_size = w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        c_end = c_start + int(chunk_size)
        canvas[r, c_start] = cpano[r, c]
        for p in range(c_start+1, c_end+1):
            canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size))

# Flip horizontally and vertically
cpano = cv2.flip(cpano, -1)
canvas = cv2.flip(canvas, -1)

# Reconstruct the southern hemisphere
for r in range(h-1, 0, -1):
    chunks = 1 + r * d
    chunk_size = w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        c_end = c_start + int(chunk_size)
        canvas[r, c_start] = cpano[r, c]
        for p in range(c_start+1, c_end+1):
            canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size))

# Flip canvas back
canvas = cv2.flip(canvas, -1)

cv2.imwrite(sys.argv[2], canvas)
