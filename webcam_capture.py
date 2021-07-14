import cv2

cam = cv2.VideoCapture(0)
cam.set(10, 160)  # Brightness
cam.set(3, 1920)  # Width
cam.set(4, 1080)  # height
cv2.namedWindow("test")

img_counter = 10

while True:
    ret, img = cam.read()
    # x, y, h, w = 700, 250, 650, 800
    # img = frame[y:y + h, x:x + w]
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", img)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "image{}.jpg".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
