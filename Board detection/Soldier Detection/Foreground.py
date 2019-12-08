import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from urllib import request
import ssl
from bs4 import BeautifulSoup

URL = "rtsp://172.29.115.197.4747/video.cgi?.mjpgh264_ulaw.sdp"
"""
Psuedo-code:
    1. take lots of picture (labeled 1-10)
    2. turn them to tensor
    3. take another picture
    4. STDIV it pixel by pixel
    5. show most dominant differences (threshold)
"""

def foreground_poc():
    threshold = 6
    newLength = 360
    newWidth = 640
    HSV = True
    std_valuesR = np.ndarray((5, newLength, newWidth), int)
    std_valuesG = np.ndarray((5, newLength, newWidth), int)
    std_valuesB = np.ndarray((5, newLength, newWidth), int)
    orig = cv2.imread("new.jpg", cv2.IMREAD_COLOR)
    if HSV:
        diff = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    [diff1, diff2, diff3] = cv2.split(diff)
    diff1 = cv2.resize(diff1, (newWidth, newLength))
    diff2 = cv2.resize(diff2, (newWidth, newLength))
    diff3 = cv2.resize(diff3, (newWidth, newLength))
    std1 = cv2.imread("std1.jpg", cv2.IMREAD_COLOR)
    if HSV:
        std1 = cv2.cvtColor(std1, cv2.COLOR_BGR2HSV)
    [std11, std12, std13] = cv2.split(std1)
    std11 = cv2.resize(std11, (newWidth, newLength))
    std12 = cv2.resize(std12, (newWidth, newLength))
    std13 = cv2.resize(std13, (newWidth, newLength))
    std2 = cv2.imread("std2.jpg", cv2.IMREAD_COLOR)
    if HSV:
        std2 = cv2.cvtColor(std2, cv2.COLOR_BGR2HSV)
    [std21, std22, std23] = cv2.split(std2)
    std21 = cv2.resize(std21, (newWidth, newLength))
    std22 = cv2.resize(std22, (newWidth, newLength))
    std23 = cv2.resize(std23, (newWidth, newLength))
    std3 = cv2.imread("std3.jpg", cv2.IMREAD_COLOR)
    if HSV:
        std3 = cv2.cvtColor(std3, cv2.COLOR_BGR2HSV)
    [std31, std32, std33] = cv2.split(std3)
    std31 = cv2.resize(std31, (newWidth, newLength))
    std32 = cv2.resize(std32, (newWidth, newLength))
    std33 = cv2.resize(std33, (newWidth, newLength))
    std4 = cv2.imread("std4.jpg", cv2.IMREAD_COLOR)
    if HSV:
        std4 = cv2.cvtColor(std4, cv2.COLOR_BGR2HSV)
    [std41, std42, std43] = cv2.split(std4)
    std41 = cv2.resize(std41, (newWidth, newLength))
    std42 = cv2.resize(std42, (newWidth, newLength))
    std43 = cv2.resize(std43, (newWidth, newLength))
    std_valuesR[0] = std11
    std_valuesG[0] = std12
    std_valuesB[0] = std13
    std_valuesR[1] = std21
    std_valuesG[1] = std22
    std_valuesB[1] = std23
    std_valuesR[2] = std31
    std_valuesG[2] = std32
    std_valuesB[2] = std33
    std_valuesR[3] = std41
    std_valuesG[3] = std42
    std_valuesB[3] = std43
    std_valuesR[4] = diff1
    std_valuesG[4] = diff2
    std_valuesB[4] = diff3
    aB = np.std(std_valuesB, axis=0)
    aG = np.std(std_valuesG, axis=0)
    aR = np.std(std_valuesR, axis=0)
    aB4 = np.std(std_valuesB[0:4], axis=0)
    aG4 = np.std(std_valuesG[0:4], axis=0)
    aR4 = np.std(std_valuesR[0:4], axis=0)
    a = np.sqrt(aB4**2 + aG4**2 + aR4**2)
    b = np.sqrt(aB**2 + aG**2 + aR**2)
    for i in range(len(a)):
        for j in range(len(a[0])):
            if b[i][j] - a[i][j] > threshold:
                a[i][j] = 1
            else:
                a[i][j] = 0
    kernel = np.ones((3, 3), np.uint8)
    #plt.imshow(orig), plt.colorbar(), plt.title("Original"), plt.show()
    #plt.imshow(a), plt.colorbar(), plt.title(threshold), plt.show()
    a = cv2.erode(a, kernel, iterations=1)
    a = cv2.dilate(a, kernel, iterations=1)
    #plt.imshow(a), plt.colorbar(), plt.title(threshold), plt.show()
    a = cv2.dilate(a, kernel, iterations=3)
    a = cv2.erode(a, kernel, iterations=3)
    #plt.imshow(a), plt.colorbar(), plt.title(threshold), plt.show()
    return a


    """
    threshold = 50
    background = cv2.imread("bg.jpg", cv2.IMREAD_GRAYSCALE)
    diff = cv2.imread("diff.jpg", cv2.IMREAD_GRAYSCALE)
    diff = cv2.resize(diff, (640, 480))
    background = cv2.resize(background, (640, 480))
    kernel = np.ones((5, 5), np.float32) / 25
    #background = cv2.filter2D(background, -1, kernel)
    #diff = cv2.filter2D(diff, -1, kernel)
    #bgdModel = np.zeros((1,65), np.float64)
    #fgdModel = np.zeros((1, 65), np.float64)
    #cv2.grabCut(diff, background, (1,1,639,479), bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    background2 = background
    plt.imshow(diff), plt.colorbar(), plt.show()
    for i in range(len(diff)):
        for j in range(len(diff[0])):
            if np.abs(int(diff[i][j]) - int(background[i][j])) < threshold:
                diff[i][j] = 0
            else:
                diff[i][j] = 1
    #background2 = np.where((np.abs(background - diff) > threshold), 1, 0)
    plt.imshow(diff), plt.colorbar(), plt.show()
    print("hello")
    """

def get_video():
    cap = cv2.VideoCapture(URL)
    while (True):
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('frame', frame)
        q = cv2.waitKey(1)
        if q == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    foreground_poc()