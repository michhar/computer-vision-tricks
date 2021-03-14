"""
HSV Colorspace plot with the following ranges:

H: 0-179 (hue range)
S: 0-255 (saturation range)
V: 255 (constant vibrance)

Use this plot to understand the desired values for the masking of
a desired color with these projects which use HSV colorspaces.

Based on: https://stackoverflow.com/questions/10787103/2d-hsv-color-space-in-matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


s, h = np.mgrid[0:1:256j, 0:1:180j]
v = np.ones_like(s)
hsv = np.dstack((h,s,v))
rgb = hsv_to_rgb(hsv)

print(rgb.shape)
plt.imshow(rgb, origin="lower", extent=[0, 180, 0, 256])
plt.xlabel("H")
plt.ylabel("S")
plt.title("$V_{HSV}=255$")
plt.show()
