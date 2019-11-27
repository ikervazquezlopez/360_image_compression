import cv2
import numpy as np




pano = cv2.imread("data/test.png")

h, w, c = pano.shape

#canvas = np.zeros_like(pano)
canvas = np.zeros( (int(h/2), w, c), dtype=np.uint8)

d = 4 # Increment rate per row

for r in range(1, int(h/2)):
#for r in range(1, 4):
    chunks = 1 + r * d
    chunk_size =  w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        canvas[r,c] = pano[r, c_start]
        pano[r, int(c_start + chunk_size/2)] = [255, 0, 0]
for r in range(1, int(h/2)):
    chunks = 1 + r * d
    chunk_size =  w / chunks
    for c in range(0, chunks):
        c_start = int(c*chunk_size)
        canvas[int(h/2)-r-1,w-c-1] = pano[h-r-1, w-c_start-1]

cv2.imwrite("out.png", canvas)
cv2.imwrite("out_pano.png", pano)
