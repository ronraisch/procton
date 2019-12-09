import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path


def calc_std(std_dir, hsv):
    std_file_count = len([name for name in os.listdir(std_dir)])
    new_length = 360
    new_width = 640
    std_values_r = np.ndarray((std_file_count + 1, new_length, new_width), int)
    std_values_g = np.ndarray((std_file_count + 1, new_length, new_width), int)
    std_values_b = np.ndarray((std_file_count + 1, new_length, new_width), int)
    count = 0
    for file in os.listdir(std_dir):
        std = cv2.imread(std_dir + '\\' + file, cv2.IMREAD_COLOR)
        plt.imshow(std)
        if hsv:
            std = cv2.cvtColor(std, cv2.COLOR_BGR2HSV)
        [std_1, std_2, std_3] = cv2.split(std)
        std_1 = cv2.resize(std_1, (new_width, new_length))
        std_2 = cv2.resize(std_2, (new_width, new_length))
        std_3 = cv2.resize(std_3, (new_width, new_length))
        std_values_r[count] = std_1
        std_values_g[count] = std_2
        std_values_b[count] = std_3
        count += 1
    mean_r = np.mean(std_values_r, axis=0)
    mean_g = np.mean(std_values_g, axis=0)
    mean_b = np.mean(std_values_b, axis=0)
    std_r = np.std(std_values_b, axis=0)
    std_g = np.std(std_values_g, axis=0)
    std_b = np.std(std_values_r, axis=0)
    cv2.imwrite("std_red.jpg", std_r)
    cv2.imwrite("std_green.jpg", std_g)
    cv2.imwrite("std_blue.jpg", std_b)
    cv2.imwrite("mean_red.jpg", mean_r)
    cv2.imwrite("mean_green.jpg", mean_g)
    cv2.imwrite("mean_blue.jpg", mean_b)


def foreground_poc(threshold, diff_path, std_dir, hsv):
    [threshold_r, threshold_g, threshold_b] = threshold
    new_length = 360
    new_width = 640
    orig = cv2.imread(diff_path, cv2.IMREAD_COLOR)
    diff = orig
    if hsv:
        diff = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    [diff_r, diff_g, diff_b] = cv2.split(diff)
    diff_r = cv2.resize(diff_r, (new_width, new_length))
    diff_g = cv2.resize(diff_g, (new_width, new_length))
    diff_b = cv2.resize(diff_b, (new_width, new_length))
    calc_std(std_dir, hsv)
    std_r = cv2.imread("std_red.jpg", 0).astype(int)
    std_g = cv2.imread("std_green.jpg", 0).astype(int)
    std_b = cv2.imread("std_blue.jpg", 0).astype(int)
    mean_r = cv2.imread("mean_red.jpg", 0).astype(int)
    mean_g = cv2.imread("mean_green.jpg", 0).astype(int)
    mean_b = cv2.imread("mean_blue.jpg", 0).astype(int)
    # np.where(std_r <= 1, 1, std_r)
    # np.where(std_g <= 1, 1, std_g)
    # np.where(std_b <= 1, 1, std_b)
    std_r[std_r < 1] = 1
    std_b[std_b < 1] = 1
    std_g[std_g < 1] = 1
    distance_r = np.abs((mean_r - diff_r))
    distance_g = np.abs((mean_g - diff_g))
    distance_b = np.abs((mean_b - diff_b))
    plt.imshow(distance_r.astype(int)), plt.title("R"), plt.colorbar(), plt.show()
    plt.imshow(distance_g.astype(int)), plt.title("G"), plt.colorbar(), plt.show()
    plt.imshow(distance_b.astype(int)), plt.title("B"), plt.colorbar(), plt.show()
    for i in range(len(distance_r)):
        for j in range(len(distance_r[0])):
            if distance_r[i][j] >= threshold_r:
                distance_r[i][j] = 1
            else:
                distance_r[i][j] = 0

            if distance_b[i][j] >= threshold_b:
                distance_b[i][j] = 1
            else:
                distance_b[i][j] = 0

            if distance_g[i][j] >= threshold_g:
                distance_g[i][j] = 1
            else:
                distance_g[i][j] = 0
    check_mat = np.zeros(distance_r.shape)
    for i in range(len(check_mat)):
        for j in range(len(check_mat[0])):
            if distance_r[i][j] == 1 and distance_b[i][j] == 1 and distance_g[i][j] == 1:
                check_mat[i][j] = 1
            else:
                check_mat[i][j] = 0
    kernel = np.ones((9,9), np.uint8)
    plt.imshow(orig), plt.title("Original"), plt.show()
    plt.imshow(check_mat), plt.colorbar(), plt.show()
    check_mat = cv2.morphologyEx(check_mat, cv2.MORPH_CLOSE, kernel, iterations=3)
    plt.imshow(check_mat), plt.colorbar(), plt.title("close"), plt.show()
    check_mat = cv2.morphologyEx(check_mat, cv2.MORPH_OPEN, kernel, iterations=5)
    plt.imshow(check_mat), plt.colorbar(), plt.title("open"), plt.show()
    return check_mat


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
