"""
Convert RGB to lower and upper limit HSV values
"""
import numpy as np
import cv2


# This is the input RGB value
# rgb = [161, 118, 89] # brown
# rgb = [255, 0, 0] # dark red
rgb = [245, 39, 242] # pink
rgb = [0, 0, 255] # blue

bgr = np.uint8([[rgb[::-1]]])
hsvColor = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
print(hsvColor)

lowerLimit = hsvColor[0][0][0] - 10, 100, 100
upperLimit = hsvColor[0][0][0] + 10, 255, 255

print(lowerLimit)
print(upperLimit)
