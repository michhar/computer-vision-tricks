"""
HSV Colorspace plot with the following ranges (these being the OpenCV 
ranges):

H: 0-179 (hue range)
S: 0-255 (saturation range)
V: 255 (value - kept constant)

Use this plot to understand the desired values for the masking of
a desired color with these projects which use HSV colorspaces.

Based on: https://stackoverflow.com/questions/10787103/2d-hsv-color-space-in-matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib
matplotlib.use('TkAgg')


def draw_hsv_colorspace():
    s, h = np.mgrid[0:1:255j, 0:1:179j]
    v = np.ones_like(s)
    hsv = np.dstack((h,s,v))
    rgb = hsv_to_rgb(hsv)

    print(rgb.shape)
    plt.imshow(rgb, origin="lower", interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()
    plt.xlabel("H")
    plt.ylabel("S")
    plt.xticks(range(0,179,20))
    plt.title("$V_{HSV}=255$")
    plt.show()

if __name__ == "__main__":
    draw_hsv_colorspace()
