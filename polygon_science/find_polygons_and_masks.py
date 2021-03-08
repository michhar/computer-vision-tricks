import cv2
import numpy as np
import argparse
from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon
import matplotlib.pyplot as plt


def blur(image):
    #calculate x & y gradient
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur the image - this is going to be sensitive to the resolution/size of image
    # so tune to your needs
    blurred = cv2.blur(gradient, (3, 3))

    return blurred

def mask_to_polygons(mask, epsilon=10., min_area=10.):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons

def mask_for_polygons(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def find_mask(image):
    """
    Mask the select color portions of the image based on HSV 
    color wheel and save mask image
    Note:  OpenCV uses  H: 0-179, S: 0-255, V: 0-255
    """
    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Range for blue
    # lower_hsv = np.array([91, 100, 100])
    # upper_hsv = np.array([111, 255, 255])
    # mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Range for red ("pinkish red") - note, there are two reddish 
    # regions on Hue scale
    lower_hsv = np.array([170,50,50])
    upper_hsv = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Generating the final mask to detect color
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             np.ones((3,3),
                             np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,
                             np.ones((3,3),
                             np.uint8))

    # creating image showing static background frame pixels only for the masked region
    res = cv2.bitwise_and(image, image, mask=mask1)

    return res

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to the image file")
    parser.add_argument(
        "--show", default=False, action="store_true",
        help="Show the steps in QR code identification"
    )

    args = parser.parse_args()    
    
    orig_image = cv2.imread(args.image)

    mask1 = find_mask(orig_image)
    plt.imsave("results/1_morph_mask.jpg", mask1[...,::-1], cmap='gray')

    # Cconvert to grayscale
    gray = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

    # Convert to black and white
    bw = cv2.convertScaleAbs(gray)
    plt.imsave("results/2_bw.jpg", bw, cmap='gray')

    # Get the polygons using shapely
    polys = mask_to_polygons(bw, min_area=100)
    print("{} polygons found.".format(len(polys)))

    # Convert the polygons back to a mask image to validate that all went well
    mask2 = mask_for_polygons(polys, orig_image.shape[:2])

    # Save - you'll see some loss in detail compared to the before-polygon 
    # image if min_area is high - go ahead and try different numbers!
    plt.imsave("results/3_polygon_mask.jpg", mask2, cmap='gray')