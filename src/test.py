import cv2
import numpy
import EquirectangularDownsampling as ed
import rhomboid_downsampling as rd


pano = cv2.imread("../data/pano1056.png", cv2.IMREAD_GRAYSCALE)
pano = cv2.resize(pano, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)



"""
================================================================================
Test split
"""

"""
left, right = ed.split(pano)
print(left.shape, right.shape)

cv2.imshow("left", left)
cv2.imshow("right", right)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
================================================================================
"""




"""
================================================================================
Test Lee Downsampling
"""
I_r = rd.rhomboid_downsample(pano)
cv2.imshow("Rhomboid downsample", I_r)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
================================================================================
"""
