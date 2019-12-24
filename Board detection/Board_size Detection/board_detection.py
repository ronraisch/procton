import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Capture frame-by-frame
ret, frame = cap.read()
# Our operations on the frame come here
while np.sum(frame) == 0 or frame is None:
    ret, frame = cap.read()
    # Our operations on the frame come here
img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
tmp = cv2.resize(img, (1140, 640))
cv2.imshow('Original Image', tmp)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 700, 900, apertureSize=5)
edges = cv2.morphologyEx(edges, kernel=None, op=cv2.MORPH_CLOSE, iterations=2)
# edges = cv2.morphologyEx(edges, kernel=None, op=cv2.MORPH_OPEN, iterations=1)

# lines=cv2.HoughLines(edges, 1, np.pi / 180, 10)
#
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(edges,(x1,y1),(x2,y2),(255,255,255),2)

tmp = cv2.resize(edges, (1140, 640))
cv2.imshow('edges Image', tmp)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tmp = img.copy()
cv2.drawContours(tmp, contours, -1, (0, 255, 0), 2)
tmp = cv2.resize(tmp, (1140, 640))
cv2.imshow('Contour Image', tmp)
cv2.waitKey(0)

contours_list = []

for contour in contours:
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    area = cv2.contourArea(contour)
    if len(approx) < 12 and area > img.shape[0] * img.shape[1] * 0.1:
        contours_list.append(contour)

biggest_contour = None
biggest_area = 0
for contour in contours_list:
    area = cv2.contourArea(contour)
    if area > biggest_area:
        biggest_area = area
        biggest_contour = contour

# cv2.drawContours(img, [biggest_contour], -1, (120, 255, 31), 2)
hull = cv2.convexHull(biggest_contour)
hull_poly = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
cv2.drawContours(img, [hull_poly], 0, (0, 255, 255), 2)

tmp = cv2.resize(img, (1140, 640))
cv2.imshow('Output Image', tmp)
cv2.waitKey(0)