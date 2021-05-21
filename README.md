# 360_image_compression


This code takes the insights from the paper "*An Analysis on Pixel Redundancy Structure in Equirectangular Images*" [REF] to vertically split the equirectangular image in two images and process each part independently to generate to romboids as in the paper "*Omnidirectional Video Coding Using Latitude Adaptive Down-Sampling and Pixel Rearrangement*" [REF].


### To Do
* Vertically split the image at latitude 0
* Downsample both parts independently and generate two romboids (one per part)
* Rearrange the romboid pixels:
  * (1) Make them into triangles and fit them in the shame hemisphere
  * (2) Use the same approach as in [REF romboids]
* Compute the quality metrics and compare against [REF romboids] results.
