import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def foreground_poc(threshold, picture_paths):
    if threshold == 0:
        threshold = 10
    new_length = 360
    new_width = 640
    hsv = True
    std_values_r = np.ndarray((5, new_length, new_width), int)
    std_values_g = np.ndarray((5, new_length, new_width), int)
    std_values_b = np.ndarray((5, new_length, new_width), int)
    orig = cv2.imread(picture_paths[0], cv2.IMREAD_COLOR)
    diff = orig
    if hsv:
        diff = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    [diff1, diff2, diff3] = cv2.split(diff)
    diff1 = cv2.resize(diff1, (new_width, new_length))
    diff2 = cv2.resize(diff2, (new_width, new_length))
    diff3 = cv2.resize(diff3, (new_width, new_length))
    std1 = cv2.imread(picture_paths[1], cv2.IMREAD_COLOR)
    if hsv:
        std1 = cv2.cvtColor(std1, cv2.COLOR_BGR2HSV)
    [std11, std12, std13] = cv2.split(std1)
    std11 = cv2.resize(std11, (new_width, new_length))
    std12 = cv2.resize(std12, (new_width, new_length))
    std13 = cv2.resize(std13, (new_width, new_length))
    std2 = cv2.imread(picture_paths[2], cv2.IMREAD_COLOR)
    if hsv:
        std2 = cv2.cvtColor(std2, cv2.COLOR_BGR2HSV)
    [std21, std22, std23] = cv2.split(std2)
    std21 = cv2.resize(std21, (new_width, new_length))
    std22 = cv2.resize(std22, (new_width, new_length))
    std23 = cv2.resize(std23, (new_width, new_length))
    std3 = cv2.imread(picture_paths[3], cv2.IMREAD_COLOR)
    if hsv:
        std3 = cv2.cvtColor(std3, cv2.COLOR_BGR2HSV)
    [std31, std32, std33] = cv2.split(std3)
    std31 = cv2.resize(std31, (new_width, new_length))
    std32 = cv2.resize(std32, (new_width, new_length))
    std33 = cv2.resize(std33, (new_width, new_length))
    std4 = cv2.imread(picture_paths[4], cv2.IMREAD_COLOR)
    if hsv:
        std4 = cv2.cvtColor(std4, cv2.COLOR_BGR2HSV)
    [std41, std42, std43] = cv2.split(std4)
    std41 = cv2.resize(std41, (new_width, new_length))
    std42 = cv2.resize(std42, (new_width, new_length))
    std43 = cv2.resize(std43, (new_width, new_length))
    std_values_r[0] = std11
    std_values_g[0] = std12
    std_values_b[0] = std13
    std_values_r[1] = std21
    std_values_g[1] = std22
    std_values_b[1] = std23
    std_values_r[2] = std31
    std_values_g[2] = std32
    std_values_b[2] = std33
    std_values_r[3] = std41
    std_values_g[3] = std42
    std_values_b[3] = std43
    std_values_r[4] = diff1
    std_values_g[4] = diff2
    std_values_b[4] = diff3
    a_b = np.std(std_values_b, axis=0)
    a_g = np.std(std_values_g, axis=0)
    a_r = np.std(std_values_r, axis=0)
    a_b4 = np.std(std_values_b[0:4], axis=0)
    a_g4 = np.std(std_values_g[0:4], axis=0)
    a_r4 = np.std(std_values_r[0:4], axis=0)
    a = np.sqrt(a_b4**2 + a_g4**2 + a_r4**2)
    b = np.sqrt(a_b**2 + a_g**2 + a_r**2)
    for i in range(len(a)):
        for j in range(len(a[0])):
            if b[i][j] - a[i][j] > threshold:
                a[i][j] = 1
            else:
                a[i][j] = 0
    kernel = np.ones((3, 3), np.uint8)
    plt.imshow(orig), plt.title("Original"), plt.show()
    # plt.imshow(std1), plt.show()
    # plt.imshow(a), plt.colorbar(), plt.title(threshold), plt.show()
    a = cv2.erode(a, kernel, iterations=2)
    a = cv2.dilate(a, kernel, iterations=4)
    # plt.imshow(a), plt.colorbar(), plt.title(threshold), plt.show()
    a = cv2.dilate(a, kernel, iterations=4)
    a = cv2.erode(a, kernel, iterations=2)
    # plt.imshow(a), plt.colorbar(), plt.title(threshold), plt.show()
    return a


"""
def get_video():
    cap = cv2.VideoCapture(URL)
    while True:
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('frame', frame)
        q = cv2.waitKey(1)
        if q == ord("q"):
            break
    cv2.destroyAllWindows()
"""
