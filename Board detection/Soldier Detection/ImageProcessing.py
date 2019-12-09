import cv2
import numpy as np
import Foreground

DEFAULF = 0
diff_path = 'new2.jpg'

img = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
print(img.shape)
# img = cv2.medianBlur(img,5)
img = cv2.resize(img, (640, 360))
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=17, param2=17, minRadius=5, maxRadius=15)

circles = np.uint16(np.around(circles))
whiteCircles = np.ndarray(shape=circles.shape)
blackCircles = np.ndarray(shape=circles.shape)

foreground = Foreground.foreground_poc(threshold=DEFAULF,diff_path=diff_path,
                                       std_dir='C:\\Users\\t8545065\\Desktop\\Lil project\\IBUUUUUD\\stds')

whiteCount = 0
blackCount = 0
for i in circles[0, :]:
    print(i)
    if foreground[i[1]][i[0]] == 1:
        if img[i[1]][i[0]] > 100:
            whiteCircles[0][whiteCount] = i
            whiteCount += 1
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)
        else:
            blackCircles[0][blackCount] = i
            blackCount += 1
            cv2.circle(cimg, (i[0], i[1]), i[2], (255, 255, 255), 2)
        # draw the outer circle
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)


cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
