import cv2
import numpy as np


def row_linear_interpolation(p, r, c, c_start, chunk_size):
    d_start = (p - c_start) / chunk_size
    d_end = 1 - d_start
    color = d_start*cpano[r, c] + d_end*cpano[r, c]
    return color.astype(np.uint8)


cpano = cv2.imread("out.jpeg")

h, w, c = cpano.shape

#canvas = np.zeros_like(pano)
canvas = np.zeros( (h*2, w, c), dtype=np.uint8)

d = 4 # Increment rate per row

for c in range(0, w-2): # Set middle row without compression
    canvas[h-1, c] = cpano[h-1, c]
    canvas[h, c] = cpano[0, c+2]

for r in range(h-2, 0, -1):
    chunks = 1 + r * d
    chunk_size = w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        c_end = c_start + int(chunk_size)
        canvas[r, c_start] = cpano[r, c]
        for p in range(c_start+1, c_end+1):
            canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size))


cpano = cv2.flip(cpano, -1)
canvas = cv2.flip(canvas, -1)

for r in range(h-2, 0, -1):
    chunks = 1 + r * d
    chunk_size = w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        c_end = c_start + int(chunk_size)
        canvas[r, c_start] = cpano[r, c]
        for p in range(c_start+1, c_end+1):
            canvas[r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size))
"""
for r in range(0, h):
    chunks = 1 + (h-r) * d
    chunk_size = w / chunks
    for c in range(w-2, w-chunks, -1):
        if r == 376:
            print(r, c, chunks, chunk_size)
        c_start = w-int(c*chunk_size)
        c_end = c_start - int(chunk_size)
        canvas[h+r, c_start] = cpano[r, c]
        for p in range(c_start, c_end, -1):
            canvas[h+r, p] = row_linear_interpolation(p, r, c, c_start, int(chunk_size))
"""
canvas = cv2.flip(canvas, -1)
cv2.imwrite("reconstructed.png", canvas)
