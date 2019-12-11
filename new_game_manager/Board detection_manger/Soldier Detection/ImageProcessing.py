import cv2
import numpy as np
import Foreground
import perspective_change


def get_detected_soldiers(filename):
    std_dir = "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\new_game_manager\\Board detection_manger\\Soldier Detection\\stds\\new_test"
    threshold_r = 1
    threshold_g = 10
    threshold_b = 4
    height = 620
    width = 650
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = perspective_change.get_perspective_change(None, img=img)

    # img = cv2.medianBlur(img,5)
    img = cv2.resize(img, (width, height))
    orig_img=cv2.imread(filename, cv2.IMREAD_COLOR)
    orig_img = perspective_change.get_perspective_change(None, img=orig_img)
    orig_img = cv2.resize(orig_img, (width, height))
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow("dsa", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 32,
                               param1=45, param2=10, minRadius=18, maxRadius=23)
    circles = np.uint16(np.around(circles))
    white_circles = np.ndarray(shape=circles.shape)
    black_circles = np.ndarray(shape=circles.shape)

    foreground = Foreground.foreground_poc(threshold=[threshold_r, threshold_g, threshold_b],
                                           diff_img=orig_img, new_width=width, new_height=height,
                                           std_dir=None)

    white_count = 0
    black_count = 0
    for i in circles[0, :]:
        if foreground[i[1]][i[0]] == 1:
            if img[i[1]][i[0]] > 100:
                white_circles[0][white_count] = i
                white_count += 1
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)
            else:
                black_circles[0][black_count] = i
                black_count += 1
                cv2.circle(cimg, (i[0], i[1]), i[2], (255, 255, 255), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return white_circles, black_circles, cimg
get_detected_soldiers(
    "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\new_game_manager\\Board detection_manger\\Soldier Detection\\stds\\new_test_std\\WIN_20191210_22_28_15_Pro.jpg")
