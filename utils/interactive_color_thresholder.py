"""
This script gives the user an interactive program to aid in choosing the
right HSV ranges that will help in the masking process.

Based on: https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
"""
import cv2
import numpy as np
import argparse


def on_trackbar_change(val):
    print(f"Trackbar value changed to: {val}")

def main(infile):
    # Load in image
    try:
        image = cv2.imread(infile)
    except Exception as e:
        print(e)
        exit()

    # Create a window
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,on_trackbar_change) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,on_trackbar_change)
    cv2.createTrackbar('VMin','image',0,255,on_trackbar_change)
    cv2.createTrackbar('HMax','image',0,179,on_trackbar_change)
    cv2.createTrackbar('SMax','image',0,255,on_trackbar_change)
    cv2.createTrackbar('VMax','image',0,255,on_trackbar_change)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image,image, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print(f"(hMin = {hMin} , sMin = {sMin}, vMin = {vMin}), (hMax = {hMax} , sMax = {sMax}, vMax = {vMax})")
            print(f'Array formats: lower_limit = [{hMin}, {sMin}, {vMin}], upper_limit = [{hMax}, {sMax}, {vMax}]')
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image',output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input image path.",
                        required=True,
                        dest='_input',
                        type=str)
    args = parser.parse_args()

    main(args._input)