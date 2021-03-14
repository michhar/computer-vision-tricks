"""
Detect QR codes in an image with OpenCV

Based on the following:
- https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/
- https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
- https://www.pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/
- https://github.com/pyxploiter/Barcode-Detection-and-Decoding

Important notes
- A transformation from RGB to HSV is performed to deal with illumination differences
"""
import cv2
import argparse
import numpy as np
import math


def find_barcode(image, orig_image, orig_box_tag, show):
    """
    Takes a straightened, cropped image of an object and finds the
    top three QR code(s) and draw bounding box, save image
    """
    #resize image
    # image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #calculate x & y gradient
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    if show:
        cv2.imshow("gradient-sub",cv2.resize(gradient,None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

    # blur the image - this is going to be sensitive to the resolution/size of image
    # so tune to your needs
    blurred = cv2.blur(gradient, (3, 3))

    # threshold the image - also may need some tuning
    (_, thresh) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)

    if show:
        cv2.imshow("threshed",cv2.resize(thresh,None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if show:
        cv2.imshow("morphology",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

    # perform a series of erosions and dilations morphological operations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    if show:
        cv2.imshow("erode/dilate",cv2.resize(closed,None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    sorted_contours = sorted(cnts, key=cv2.contourArea, reverse=True)

    # show top three contours/boxes by size
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    cnt = 0
    for i in range(len(sorted_contours)):
        # compute the rotated bounding box of the largest contour
        # convert from the contour to a box
        # the box dims are:  [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        # where the points start at the lowest point and go clockwise
        cont = sorted_contours[i]
        rect1 = cv2.minAreaRect(cont)
        box1 = np.int0(cv2.boxPoints(rect1))

        # draw a bounding box arounded the detected barcode and display the
        # image (and three largest boxes)
        cv2.drawContours(image, [box1], -1, colors[i], 4) # Red box (biggest)
        
        if cnt == 2:
            break
        cnt+=1

    # Show barcode id
    cv2.imshow("barcode_id", image)
    cv2.imwrite("results/qrcode_found_straight.jpg", image)

    # Rotation of cropped image above back to orig rotation

    # perspective transform and warp
    height, width = image.shape[0], image.shape[1]
    src_pts = np.array([[0, height-1],
                         [0, 0],
                         [width-1, 0],
                         [width-1, height-1]], dtype="float32")
    # coordinate of points in box after the rect has been straightened
    dst_pts = np.array(orig_box_tag, dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the straightened rectangle image to get the rotated rectangle
    tag_warped = cv2.warpPerspective(image, M, (orig_image.shape[1], orig_image.shape[0]))

    alpha = 0.5
    beta = 1.0 - alpha
    # # add the rotated rectangle back to original image
    result = cv2.addWeighted(orig_image, alpha, tag_warped, beta, 0.0)

    cv2.imshow("barcode_id_orig", result)
    cv2.imwrite("results/qrcodes_found_orig_image.jpg", result)


def angle(vector1, vector2):
    """Return the angle between two vectors"""
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))

def find_mask(image):
    """Mask the select color portions of the image based on HSV 
    color wheel and save mask image"""
    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Range for blue
    # lower_hsv = np.array([91, 100, 100])
    # upper_hsv = np.array([111, 255, 255])
    # mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # # Range for brown
    # lower_hsv = np.array([5, 100, 100])
    # upper_hsv = np.array([25, 255, 255])
    # mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Range for white
    lower_hsv = np.array([0,0,0])
    upper_hsv = np.array([255,10,255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  

    # Generating the final mask to detect color
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    # creating image showing static background frame pixels only for the masked region
    res = cv2.bitwise_and(image, image, mask=mask1)

    #Generating the final output
    cv2.imwrite("results/color_masked.jpg", res)

    return mask1

def find_tag_box(mask, orig_image):
    """Find the bounding box around the largest specifically colored
    object/contour and save image"""
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

    sorted_contours = sorted(cnts, key = cv2.contourArea, reverse=True)
    if len(sorted_contours) > 0:
        c1 = sorted_contours[0] # biggest by area
    else:
        return [], []

    # compute the rotated bounding box of the largest contour
    rect1 = cv2.minAreaRect(c1)
    box1 = np.int0(cv2.boxPoints(rect1))

    # draw a bounding box arounded the detected area and display the
    # image (and largest box)
    res_image = orig_image.copy()
    cv2.drawContours(res_image, [box1], -1, (0, 0, 255), 5) # Red box (biggest)

    cv2.imwrite("results/color_boxed.jpg", res_image)

    return rect1, box1

def straighten_and_crop(rect, box, orig_image):
    """
    Straighten image based on a region box and
    crop the tag region, box, from the image, return cropped
    
    This will "pad" image with black background as rotating so as
    to not truncate the boxed tag region
    """
    # get width and height for cropping
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of points in box after the rect has been straightened
    dst_pts = np.array([[0, height-1],
                         [0, 0],
                         [width-1, 0],
                         [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    cropped = cv2.warpPerspective(orig_image, M, (width, height))

    # cv2.imshow("cropped", cropped)
    cv2.imwrite("results/color_cropped.jpg", cropped)

    return cropped

def main(args):
    """Main function"""

    # Laterally invert the image / flip the image
    orig_image = cv2.imread(args.image)

    # Run algo to mask a part of image based on color
    mask1 = find_mask(orig_image)

    # Run algo to find the prominent object and bounding box
    rect, box = find_tag_box(mask1, orig_image)

    if len(rect) > 0 and len(box) > 0:
        # Run algo to straighten identified box and crop it
        cropped = straighten_and_crop(rect, box, orig_image)

        # Find barcode in straightened, cropped image and save image
        find_barcode(cropped, orig_image, box, args.show)

    print("Ctrl-C to exit")

	# To keep images on screen, loop to detect when to quit program
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to the image file")
    parser.add_argument(
        "--show", default=False, action="store_true",
        help="Show the steps in QR code identification"
    )

    args = parser.parse_args()    
    main(args)