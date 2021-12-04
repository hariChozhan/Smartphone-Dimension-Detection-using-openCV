from time import sleep
import cv2
from numpy import ones, array
from imutils import grab_contours
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

debug = True
# inputImage = 'testPhone2.jpg'  # Image to be measured
ppm = 57.33557172040636  # Pixel Per Metric

cam = cv2.VideoCapture(0)
cam.set(10, 160)  # Brightness
cam.set(3, 1920)  # Width
cam.set(4, 1080)  # height
# cv2.namedWindow("test")

while True:
    ret, img = cam.read()
    # x, y, h, w = 700, 250, 650, 800
    # img = frame[y:y + h, x:x + w]
    if not ret:
        print("failed to grab frame")
        break
    # cv2.imshow("test", img)
    sleep(5)
    cv2.imwrite('capturedImage.jpg', img)

    inputImage = 'capturedImage.jpg'
    imgOriginal = cv2.imread(inputImage)
    if debug: cv2.imshow("Original Image", imgOriginal)
    # x, y, h, w = 210, 110, 300, 295
    # imgCrop = imgOriginal[y:y + h, x:x + w]
    # cv2.imshow("Cropped Image", imgCrop)
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGRA2GRAY)
    if debug: cv2.imshow("GrayScale Image", imgGray)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 5)
    if debug: cv2.imshow("Blur Image", imgBlur)
    imgCanny = cv2.Canny(imgBlur, 50, 200)
    if debug: cv2.imshow("Canny Image", imgCanny)
    kernel = ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    if debug: cv2.imshow('DilateImage', imgDial)
    imgErode = cv2.erode(imgDial, kernel, iterations=3)
    if debug: cv2.imshow('ErodeImage', imgErode)

    cntrs = cv2.findContours(imgErode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = grab_contours(cntrs)
    (cntrs, _) = contours.sort_contours(cntrs)
    if debug: print("Contours : ", cntrs)
    for c in cntrs:
        if debug: print("Contour Area : ", cv2.contourArea(c))
        if cv2.contourArea(c) < 1000:
            continue
        outputImage = imgOriginal.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = array(box, dtype="int")
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
        ppm = lengthPixel / 15
        print(ppm)
        length = lengthPixel / ppm
        width = widthPixel / ppm
        print("Length in cm : ", length)
        print("Width in cm : ", width)
        cv2.imshow("OUTPUT IMAGE", outputImage)
    cv2.waitKey(0)

cam.release()
cv2.destroyAllWindows()
