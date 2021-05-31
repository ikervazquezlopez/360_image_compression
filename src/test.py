import cv2
import numpy
import equirectangular_downsampling as ed
import rhomboid_downsampling as rd
import triangular_downsampling as td


pano = cv2.imread("../data/pano1056.png")
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

"""
I_r = rd.rhomboid_downsample(pano)
#I_erp = rd.rhomboid_reconstruction(I_r)
cv2.imshow("Rhomboid downsample", I_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
================================================================================
"""




"""
================================================================================
Test Rhomboid Downsampling (1,1)
"""

"""
left, right = ed.split(pano)
I_r = rd.rhomboid_downsample_1_1(left)
I_erp = rd.rhomboid_reconstruction_1_1(I_r)
I_erp = cv2.flip(I_erp, 1)
cv2.imshow("left", left)
cv2.imshow("Rhomboid downsample", I_erp)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
================================================================================
"""




"""
================================================================================
Test Triangular Downsampling (2,1)
"""


I_t = td.triangular_downsampling_2_1(pano)
I_erp = td.triangular_reconstruction_2_1(I_t)
cv2.imshow("Triangular reconstruction", I_erp)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("out.png", I_erp)


"""
================================================================================
"""





"""
================================================================================
Test Triangular Downsampling (1,1)
"""

"""
left, right = ed.split(pano)
I_t = td.triangular_downsampling_1_1(left)
#I_erp = rd.rhomboid_reconstruction_1_1(I_r)
#I_erp = cv2.flip(I_erp, 1)
cv2.imshow("Triangular downsample", I_t)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("left.png", I_t)
"""

"""
================================================================================
"""
