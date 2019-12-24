import numpy as np
import cv2

RESIZE_WIDTH = 1140
RESIZE_HEIGHT = 640

MIN_RADIUS = 25
MAX_RADIUS = 50
MIN_DIST = 60
DIST_THRESHOLD = 20


def define_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def take_picture(cap=define_camera()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    while np.sum(frame) == 0:
        ret, frame = cap.read()
        # Our operations on the frame come here

    img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    return img


def show_img(img, title="img"):
    tmp = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
    cv2.imshow(title, tmp)
    cv2.waitKey(0)


def separate_img_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(hsv)
    return cv2.split(hsv)


def mask_range(img, lower, upper):
    return cv2.inRange(img, lower, upper)


def close_img(img, iterations=3):
    """
    :param img: img that should be closed (close filter)
    :param iterations: number of times that the filter is applied
    """
    return cv2.morphologyEx(img, kernel=None, op=cv2.MORPH_CLOSE, iterations=iterations)


def open_img(img, iterations=3):
    """
    :param img: img that should be opened (open filter)
    :param iterations: number of times that the filter is applied
    """
    return cv2.morphologyEx(img, kernel=None, op=cv2.MORPH_OPEN, iterations=iterations)


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def get_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_hough_circles(gray, param1=40, param2=13):
    return list(cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=MIN_DIST,
                                 param1=param1, param2=param2, minRadius=MIN_RADIUS,
                                 maxRadius=MAX_RADIUS)[0, :])


def get_contours(gray):
    contours, _ = cv2.findContours(gray, mode=cv2.RETR_CCOMP,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img, contours, color=(189, 183, 107)):
    # cv2.drawContours(img, contours, -1, (189, 183, 107), 2)
    cv2.drawContours(img, contours, -1, color, 2)


def get_contour_circles(contours):
    """
    This function returns all the circles (with radius in wanted range) that surrounds a contour
    :param contours: the contours to search in
    :return: all the circles that where found
    """
    contour_circles = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if MIN_RADIUS < radius < MAX_RADIUS:
            contour_circles.append(np.array([x, y, radius]))
            # cv2.circle(img, (int(x), int(y)), int(radius), color=(0, 255, 0), thickness=2)
    return contour_circles


def draw_circles(img, circles, color=(0, 255, 0)):
    for circle in circles:
        cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), color, 2)


def main():
    img = take_picture()
    show_img(img, "Original image")

    # hsv is the conversion of RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([70, 0, 0])
    upper_black = np.array([150, 160, 100])
    black_mask = mask_range(hsv, lower_black, upper_black)

    lower_blue = np.array([110, 50, 75])
    upper_blue = np.array([130, 255, 255])
    blue_mask = mask_range(hsv, lower_blue, upper_blue)
    blue_mask = close_img(blue_mask, 3)

    # total_mask=black and not blue
    total_mask = np.multiply(black_mask, 255 - blue_mask)
    res_black = apply_mask(img, total_mask)
    res_black = open_img(res_black, 3)
    show_img(res_black, "Color mask Image")

    gray = get_gray_scale(res_black)
    hough_circles = get_hough_circles(gray)

    contours = get_contours(gray)
    contour_circles = get_contour_circles(contours)
    draw_contours(img, contours)

    # finding the circles that contour_circles and hough_circles agree on
    confirmed_circles = []
    for contour_circle in contour_circles:
        for hough_circle in hough_circles:
            if np.linalg.norm(contour_circle[:2] - hough_circle[:2]) < DIST_THRESHOLD:
                new_radius = max(contour_circle[2], hough_circle[2])
                tmp_circle = contour_circle.copy()
                tmp_circle[2] = new_radius
                confirmed_circles.append(tmp_circle)

    # looking for circles from hough_circles that are inside a contour
    for circle in hough_circles:
        if MIN_RADIUS < circle[2] < MAX_RADIUS:
            for contour in contours:
                if cv2.pointPolygonTest(contour, (circle[0], circle[1]), True) > 0:
                    # cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    new_circle = True
                    for confirmed_circle in confirmed_circles:
                        if np.linalg.norm(confirmed_circle[:2] - circle[:2]) < MIN_DIST:
                            confirmed_circle[2] = max(confirmed_circle[2], circle[2])
                            new_circle = False
                            break
                    if new_circle:
                        confirmed_circles.append(circle)
    draw_circles(img, confirmed_circles)
    show_img(img, "detected soldiers Image")


def get_result():
    img = take_picture()

    # hsv is the conversion of RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([70, 0, 0])
    upper_black = np.array([150, 160, 100])
    black_mask = mask_range(hsv, lower_black, upper_black)

    lower_blue = np.array([110, 50, 75])
    upper_blue = np.array([130, 255, 255])
    blue_mask = mask_range(hsv, lower_blue, upper_blue)
    blue_mask = close_img(blue_mask, 3)

    # total_mask=black and not blue
    total_mask = np.multiply(black_mask, 255 - blue_mask)
    res_black = apply_mask(img, total_mask)
    res_black = open_img(res_black, 3)

    gray = get_gray_scale(res_black)
    hough_circles = get_hough_circles(gray)

    contours = get_contours(gray)
    contour_circles = get_contour_circles(contours)

    # finding the circles that contour_circles and hough_circles agree on
    confirmed_circles = []
    for contour_circle in contour_circles:
        for hough_circle in hough_circles:
            if np.linalg.norm(contour_circle[:2] - hough_circle[:2]) < DIST_THRESHOLD:
                new_radius = max(contour_circle[2], hough_circle[2])
                tmp_circle = contour_circle.copy()
                tmp_circle[2] = new_radius
                confirmed_circles.append(tmp_circle)

    # looking for circles from hough_circles that are inside a contour
    for circle in hough_circles:
        if MIN_RADIUS < circle[2] < MAX_RADIUS:
            for contour in contours:
                if cv2.pointPolygonTest(contour, (circle[0], circle[1]), True) > 0:
                    # cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                    new_circle = True
                    for confirmed_circle in confirmed_circles:
                        if np.linalg.norm(confirmed_circle[:2] - circle[:2]) < MIN_DIST:
                            confirmed_circle[2] = max(confirmed_circle[2], circle[2])
                            new_circle = False
                            break
                    if new_circle:
                        confirmed_circles.append(circle)
    return confirmed_circles
main()


#### NOT USED FUNCTIONS ###

# def get_circle_color(x, y, radius, img):
#     circle_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#     cv2.circle(circle_img, (x, y), radius, (255, 255, 255), -1)
#     return np.array(cv2.mean(img, mask=circle_img)[::-1][1:])
#
#
# def check_circle_range(x, y, radius, img, min_color, max_color, hsv):
#     if hsv:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     color = get_circle_color(x, y, radius, img)
#     print(min_color, color)
#
#     return min_color.all() < color.all() < max_color.all()


# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#
# # Capture frame-by-frame
# ret, frame = cap.read()
# # Our operations on the frame come here
# while np.sum(frame) == 0:
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#
# img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# hue, saturation, value = cv2.split(hsv)

# tmp = cv2.resize(img, (1140, 640))
# cv2.imshow('Original Image', tmp)
# cv2.waitKey(0)

# bottom_black = 0
# top_black = 100
#
# lower_black = np.array([70, 0, bottom_black])
# upper_black = np.array([150, 160, top_black])
# black_mask = cv2.inRange(hsv, lower_black, upper_black)

# lower_blue = np.array([110, 50, 75])
# upper_blue = np.array([130, 255, 255])
# blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# blue_mask = cv2.morphologyEx(blue_mask, kernel=None, op=cv2.MORPH_CLOSE, iterations=5)

# tmp = cv2.resize(blue_mask, (1140, 640))
# cv2.imshow('Blue Image', tmp)
# cv2.waitKey(0)

# res_black = cv2.bitwise_and(img, img, mask=np.multiply(black_mask, 255 - blue_mask))
# res_black = cv2.morphologyEx(res_black, kernel=None, op=cv2.MORPH_OPEN, iterations=3)
# res_black = cv2.morphologyEx(res_black, kernel=None, op=cv2.MORPH_CLOSE, iterations=7)

# tmp = cv2.resize(res_black, (1140, 640))
# cv2.imshow('Black Image', tmp)
# cv2.waitKey(0)


# gray = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=MIN_DIST,
#                            param1=40, param2=13, minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)[0, :]
# circles = list(circles)

# for circle in circles:
#     if MIN_RADIUS < circle[2] < MAX_RADIUS and check_circle_range(circle[0], circle[1], circle[2],
#                                                                   img, lower_black,
#                                                                   upper_black):
#         cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)


# res_black=cv2.dilate(res_black,None,iterations=3)
# res_black = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
# contours, hierarchy = cv2.findContours(res_black, mode=cv2.RETR_CCOMP,
#                                        method=cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (189, 183, 107), 2)

# print(circles)


# contour_circles = []
# for contour in contours:
#     (x, y), radius = cv2.minEnclosingCircle(contour)
#     if MIN_RADIUS < radius < MAX_RADIUS:
#         contour_circles.append(np.array([x, y, radius]))
#         # cv2.circle(img, (int(x), int(y)), int(radius), color=(0, 255, 0), thickness=2)

# confirmed_circles = []
# for contour_circle in contour_circles:
#     for hough_circle in circles:
#         if np.linalg.norm(contour_circle[:2] - hough_circle[:2]) < DIST_THRESHOLD:
#             new_radius = max(contour_circle[2], hough_circle[2])
#             tmp_circle = contour_circle.copy()
#             tmp_circle[2] = new_radius
#             confirmed_circles.append(tmp_circle)
#
# for circle in circles:
#     if MIN_RADIUS < circle[2] < MAX_RADIUS:
#         for contour in contours:
#             if cv2.pointPolygonTest(contour, (circle[0], circle[1]), True) > 0:
#                 # cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
#                 new_circle = True
#                 for confirmed_circle in confirmed_circles:
#                     if np.linalg.norm(confirmed_circle[:2] - circle[:2]) < MIN_DIST:
#                         confirmed_circle[2] = max(confirmed_circle[2], circle[2])
#                         new_circle = False
#                         break
#                 if new_circle:
#                     confirmed_circles.append(circle)


# for circle in confirmed_circles:
#     cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), 2)

# tmp = cv2.resize(img, (1140, 640))
# cv2.imshow('Black Image', tmp)
# cv2.waitKey(0)