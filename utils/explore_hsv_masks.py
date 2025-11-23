"""
Identifying red regions/masks in image

Source:  https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
"""
import cv2
import numpy as np
import argparse


def main(infile):
    #blurring and smoothin
    img1=cv2.imread(infile, 1)

    hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

    # #lower red
    # lower_red = np.array([0,50,50])
    # upper_red = np.array([10,255,255])

    # #upper red
    # lower_red2 = np.array([170,50,50])
    # upper_red2 = np.array([180,255,255])

    # blue (lower)
    lower_limit1 = np.array([50,50,0])
    upper_limit1 = np.array([255,255,10])

    # rgb = [245, 39, 242] # pink
    rgb = [32, 133, 214] # blueish

    bgr = np.uint8([[rgb[::-1]]])
    hsvColor = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_limit2 = np.array([hsvColor[0][0][0] - 10, 50, 50])
    upper_limit2 = np.array([hsvColor[0][0][0] + 10, 255, 255])

    while(1):
        mask = cv2.inRange(hsv, lower_limit1, upper_limit1)
        res = cv2.bitwise_and(img1,img1, mask= mask)

        mask2 = cv2.inRange(hsv, lower_limit2, upper_limit2)
        res2 = cv2.bitwise_and(img1,img1, mask= mask2)

        img3 = res+res2
        img4 = cv2.add(res,res2)
        img5 = cv2.addWeighted(res,0.5,res2,0.5,0)


        kernel = np.ones((15,15),np.float32)/225
        smoothed = cv2.filter2D(res,-1,kernel)
        smoothed2 = cv2.filter2D(img3,-1,kernel)

        cv2.imshow('Original',img1)
        cv2.imshow('Averaging',smoothed)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        cv2.imshow('mask2',mask2)
        cv2.imshow('res2',res2)
        cv2.imshow('res3',img3)
        cv2.imshow('res4',img4)
        cv2.imshow('res5',img5)
        cv2.imshow('smooth2',smoothed2)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(1) & 0xFF == ord('q'):
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