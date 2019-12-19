import numpy as np
import cv2


def get_circle_color(x, y, radius, img):
    circle_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.circle(circle_img, (x, y), radius, (255, 255, 255), -1)
    return np.array(cv2.mean(img, mask=circle_img)[::-1][1:])


def check_circle_range(x, y, radius, img, min_color, max_color, hsv):
    if hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = get_circle_color(x, y, radius, img)
    print(min_color, color)

    return min_color.all() < color.all() < max_color.all()


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Capture frame-by-frame
ret, frame = cap.read()
# Our operations on the frame come here
while np.sum(frame) == 0:
    ret, frame = cap.read()
    # Our operations on the frame come here

img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)

tmp = cv2.resize(img, (1140, 640))
cv2.imshow('Original Image', tmp)
cv2.waitKey(0)

bottom_black = 0
top_black = 130

lower_black = np.array([0, 0, bottom_black])
upper_black = np.array([255, 255, top_black])
black_mask = cv2.inRange(hsv, lower_black, upper_black)

lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 100, 100])
blue_mask = cv2.inRange(img, lower_blue, upper_blue)
blue_mask = cv2.morphologyEx(blue_mask, kernel=None, op=cv2.MORPH_CLOSE, iterations=3)

tmp = cv2.resize(blue_mask, (1140, 640))
cv2.imshow('Blue Image', tmp)
cv2.waitKey(0)

res_black = cv2.bitwise_and(img, img, mask=np.multiply(black_mask, 255 - blue_mask))
res_black = cv2.morphologyEx(res_black, kernel=None, op=cv2.MORPH_OPEN, iterations=3)
# res_black = cv2.morphologyEx(res_black, kernel=None, op=cv2.MORPH_CLOSE, iterations=7)

tmp = cv2.resize(res_black, (1140, 640))
cv2.imshow('Black Image', tmp)
cv2.waitKey(0)

min_radius = 25
max_radius = 50
min_dist = 60
gray = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                           param1=40, param2=13, minRadius=min_radius, maxRadius=max_radius)[0, :]
circles = list(circles)

# for circle in circles:
#     if min_radius < circle[2] < max_radius and check_circle_range(circle[0], circle[1], circle[2],
#                                                                   img, lower_black,
#                                                                   upper_black):
#         cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

tmp = cv2.resize(res_black, (1140, 640))
cv2.imshow('Black Image', tmp)
cv2.waitKey(0)

# res_black=cv2.dilate(res_black,None,iterations=3)
res_black = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(res_black, mode=cv2.RETR_CCOMP,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (189, 183, 107), 2)

# print(circles)


contour_circles = []
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    if min_radius < radius < max_radius:
        contour_circles.append(np.array([x, y, radius]))
        # cv2.circle(img, (int(x), int(y)), int(radius), color=(0, 255, 0), thickness=2)

dist_threshold = 20
confirmed_circles = []
for contour_circle in contour_circles:
    for hough_circle in circles:
        if np.linalg.norm(contour_circle[:2] - hough_circle[:2]) < dist_threshold:
            confirmed_circles.append(contour_circle)

for circle in circles:
    if min_radius < circle[2] < max_radius:
        for contour in contours:
            if cv2.pointPolygonTest(contour, (circle[0], circle[1]), True) > 0:
                # cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                new_circle = True
                for confirmed_circle in confirmed_circles:
                    if np.linalg.norm(confirmed_circle[:2] - circle[:2]) < min_dist:
                        new_circle = False
                        break
                if new_circle:
                    confirmed_circles.append(circle)

for circle in confirmed_circles:
    cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), 2)

tmp = cv2.resize(img, (1140, 640))
cv2.imshow('Black Image', tmp)
cv2.waitKey(0)
