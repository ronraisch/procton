import cv2
import numpy as np
import Foreground
import Hough_circle

img = cv2.imread('new.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)
# img = cv2.medianBlur(img,5)
img = cv2.resize(img, (640, 360))
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,10,
                            param1=17, param2=17, minRadius=6, maxRadius=14)

circles = np.uint16(np.around(circles))

foreground = Foreground.foreground_poc()

for i in circles[0,:]:
    print(i)
    if foreground[i[1]][i[0]] == 1:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()