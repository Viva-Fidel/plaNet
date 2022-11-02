import cv2
import numpy as np

lower_blue = np.array([0, 0, 0])
upper_blue = np.array([80, 255, 255])

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

lower = np.array([60, 0, 0])
upper = np.array([130, 255, 255])

a = cv2.cvtColor(cv2.imread('Day 5.jpeg'), cv2.COLOR_BGR2HSV)

while True:
    #img_square = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    mask_blue = cv2.inRange(a, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = len)
        # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
        #print((w, h))

        # Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
        #(mask_blue, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(mask_blue, [box], True, (255, 0, 0), 2)

    blue = cv2.inRange(a, lower, upper)
    contours2, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt2 = max(contours2, key=len)
    # Get rect
    rect = cv2.minAreaRect(cnt2)
    (c, v), (b, n), angle = rect

    print((w, h)[0]/(b, n)[0])
    print((w, h)[1] / (b, n)[1])


    cv2.imshow("plant", mask_blue)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break