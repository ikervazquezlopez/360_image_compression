# 360_image_compression


This code takes the insights from the paper "*An Analysis on Pixel Redundancy Structure in Equirectangular Images*" [1] to vertically split the equirectangular image in two images and process each part independently to generate to romboids as in the paper "*Omnidirectional Video Coding Using Latitude Adaptive Down-Sampling and Pixel Rearrangement*" [2].


### To Do
- [X] Vertically split the image at latitude 0
- [X] Downsample both parts independently and generate two romboids (one per part)
- [ ] Rearrange the romboid pixels:
  - [] (1) Make them into triangles and fit them in the shame hemisphere
  - [X] (2) Use the same approach as in [2]
- [ ] Compute the quality metrics and compare against [2] results.




### References
[1]: To be published in WSCG 2021

[2]: Lee, S.-H., Kim, S.-T., Yip, E., Choi, B.-D., Song, J., & Ko, S.-J. (2017). Omnidirectional video coding using latitude adaptive down-sampling and pixel rearrangement. Electronics Letters, 53(10), 655–657.
