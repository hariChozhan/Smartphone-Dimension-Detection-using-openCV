import cv2
import numpy as np
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

debug = False
inputImage = 'testPhone2.jpg'  # Image to be measured
ppm = 26.64395281788317  # Pixel Per Metric

while True:
    imgOriginal = cv2.imread(inputImage)
    if debug: cv2.imshow("Original Image", imgOriginal)
    # x, y, h, w = 210, 110, 300, 295
    # imgCrop = imgOriginal[y:y + h, x:x + w]
    # cv2.imshow("Cropped Image", imgCrop)
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 5)
    if debug: cv2.imshow("Blur Image", imgBlur)
    imgCanny = cv2.Canny(imgBlur, 50, 200)
    if debug: cv2.imshow("Canny Image", imgCanny)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    if debug: cv2.imshow('DilateImage', imgDial)
    imgErode = cv2.erode(imgDial, kernel, iterations=3)
    if debug: cv2.imshow('ErodeImage', imgErode)

    cntrs = cv2.findContours(imgErode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)
    (cntrs, _) = contours.sort_contours(cntrs)
    if debug: print("Contours : ", cntrs)
    for c in cntrs:
        if debug: print("Contour Area : ", cv2.contourArea(c))
        if cv2.contourArea(c) < 1:
            continue
        outputImage = imgOriginal.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        if debug: print("Bounding Box : ", box)
        cv2.drawContours(outputImage, [box.astype("int")], 0, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(outputImage, (int(x), int(y)), 5, (0, 0, 255), -1)
        topLeft, topRight, bottomRight, bottomLeft = box
        topLeftX, topLeftY = topLeft
        topRightX, topRightY = topRight
        bottomRightX, bottomRightY = bottomRight
        bottomLeftX, bottomLeftY = bottomLeft
        if debug: print("Top Left-X : ", topLeftX, "; Top Left-Y : ", topLeftY)
        if debug: print("Top Right-X : ", topRightX, "; Top Right-Y : ", topRightY)
        if debug: print("Bottom Left-X : ", bottomLeftX, "; Bottom Left-X : ", bottomLeftY)
        if debug: print("Bottom Right-X : ", bottomRightX, "; Bottom Right-Y : ", bottomRightY)
        lengthPixel = dist.euclidean((topLeftX, topLeftY), (bottomLeftX, bottomLeftY))
        if debug: print("length in Pixels : ", lengthPixel)
        widthPixel = dist.euclidean((topLeftX, topLeftY), (topRightX, topRightY))
        if debug: print("Width in Pixels : ", widthPixel)
        # ppm = lengthPixel / 14.15
        # print(ppm)
        length = lengthPixel / ppm
        width = widthPixel / ppm
        print("Length in cm : ", length)
        print("Width in cm : ", width)
        cv2.imshow("OUTPUT IMAGE", outputImage)
    cv2.waitKey(0)
