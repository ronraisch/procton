import cv2
import matplotlib.pyplot as plt
import numpy as np
import Foreground_board
import vectorHelping

def get_corners():
    only_board_path = "C:\\Users\\t8545065\\Desktop\\Lil project\\procton\\Board detection\\Board_size Detection\\board.jpg"
    background_path = "C:\\Users\\t8545065\\Desktop\\Lil project\\procton\\Board detection\\Board_size Detection\\emptyBackgrounds"
    threshold = [0, 0, 35]
    foreground = Foreground_board.foreground_poc(threshold, only_board_path, background_path, hsv=True)
    # kernel = np.ones((2, 2), np.uint8)
    # cv2.dilate(src=foreground, kernel=kernel, iterations=10)
    plt.imshow(foreground), plt.colorbar(), plt.show()
    corners = np.zeros((4, 2))  # topLeftX, topLeftY -> clockwise
    corners[0][1] = 20000
    corners[3][0] = 20000
    for j in range(len(foreground[0])):
        for i in range(len(foreground)):
            if foreground[i][j] == 1:
                if i + j < corners[0][0] + corners[0][1]:
                    corners[0][0] = i
                    corners[0][1] = j
                if i + j > corners[2][0] + corners[2][1]:
                    corners[2][0] = i
                    corners[2][1] = j

    left_right = True
    for j in range(len(foreground[0])):
        for i in range(len(foreground)):
            if foreground[i][j] == 1:
                if j < corners[0]:
                    left_right = False
                if left_right:
                    if i < corners[3][0]:
                        corners[3][0] = i
                        corners[3][1] = j
                    if i > corners[1][0]:
                        corners[1][0] = i
                        corners[1][1] = j
                else:
                    if j > corners[1][1]:
                        corners[1][0] = i
                        corners[1][1] = j
                    if j < corners[3][1]:
                        corners[3][0] = i
                        corners[3][1] = j


    for i in range(4):
        cv2.circle(foreground, (corners[i][1], corners[i][0]), 10, (2, 2, 2), 2)
        foreground[corners[i][0]][corners[i][1]] = 2
    plt.imshow(foreground), plt.colorbar(), plt.show()
    return corners
