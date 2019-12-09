import cv2
import matplotlib.pyplot as plt
import numpy as np
import Foreground


picture_paths = ["board.jpg", "empty1.jpg", "empty2.jpg", "empty3.jpg", "empty4.jpg"]
foreground = Foreground.foreground_poc(26, picture_paths)
kernel = np.ones((2, 2), np.uint8)
cv2.dilate(src=foreground, kernel=kernel, iterations=10)
plt.imshow(foreground), plt.colorbar(), plt.show()
corners = [0, 2000000, 0, 0, 0, 0, 20000000, 0]  # topLeftX, topLeftY -> clockwise
for j in range(len(foreground[0])):
    for i in range(len(foreground)):
        if foreground[i][j] == 1:
            if i + j < corners[0] + corners[1]:
                corners[0] = i
                corners[1] = j
            if i + j > corners[4] + corners[5]:
                corners[4] = i
                corners[5] = j

left_right = True
for j in range(len(foreground[0])):
    for i in range(len(foreground)):
        if foreground[i][j] == 1:
            if j < corners[0]:
                left_right = False
            if left_right:
                if i < corners[6]:
                    corners[6] = i
                    corners[7] = j
                if i > corners[2]:
                    corners[2] = i
                    corners[3] = j
            else:
                if j > corners[3]:
                    corners[2] = i
                    corners[3] = j
                if j < corners[7]:
                    corners[6] = i
                    corners[7] = j


for i in range(4):
    cv2.circle(foreground, (corners[2*i + 1], corners[2*i]), 10, (2, 2, 2), 2)
    foreground[corners[2*i]][corners[2*i + 1]] = 2
plt.imshow(foreground), plt.colorbar(), plt.show()
print(corners)
