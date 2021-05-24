import cv2
import numpy
import EquirectangularDownsampling as ed


pano = cv2.imread("../data/pano1056.png", cv2.IMREAD_GRAYSCALE)



"""
Test split
"""
left, right = ed.split(pano)
print(left.shape, right.shape)

cv2.imshow("left", left)
cv2.imshow("right", right)
cv2.waitKey(0)
cv2.destroyAllWindows()
