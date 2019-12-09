import cv2
import numpy as np
import Foreground

def get_detected_soldiers(filename):
    DEFAULF = 0
    std_dir = "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\Board detection\\Soldier Detection\\stds"
    threshold_r = 0
    threshold_g = 6
    threshold_b = 6
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    # img = cv2.medianBlur(img,5)
    img = cv2.resize(img, (640, 360))
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 15,
                               param1=40, param2=13, minRadius=7, maxRadius=13)

    circles = np.uint16(np.around(circles))
    whiteCircles = np.ndarray(shape=circles.shape)
    blackCircles = np.ndarray(shape=circles.shape)

    foreground = Foreground.foreground_poc(threshold=[threshold_r, threshold_g, threshold_b],diff_path=filename,
                                          std_dir= std_dir)

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
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)


    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

get_detected_soldiers("C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\Board detection\\Soldier Detection\\test_photos\WIN_20191209_17_01_19_Pro.jpg")