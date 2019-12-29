import cv2
import numpy as np

RESIZE_WIDTH = 1140
RESIZE_HEIGHT = 640
AREA_THRESHOLD = 0.1


def set_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def take_picture(cap=set_camera()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    while np.sum(frame) == 0 or frame is None:
        ret, frame = cap.read()
        # Our operations on the frame come here
    return cv2.cvtColor(frame, cv2.IMREAD_COLOR)


def show_img(img, title="img"):
    tmp = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
    cv2.imshow(title, tmp)
    cv2.waitKey(0)


def get_gray_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_edges(gray=None, img=None):
    if gray is None:
        gray = get_gray_image(img)
    edges = cv2.Canny(gray, 550, 750, apertureSize=5)
    edges = cv2.morphologyEx(edges, kernel=None, op=cv2.MORPH_CLOSE, iterations=2)
    return edges


def get_contours(edges):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_draw_contours(img, contours):
    tmp = img.copy()
    cv2.drawContours(tmp, contours, -1, (0, 255, 0), 2)
    return tmp


def get_relevant_contours(contours, img):
    contours_list = []

    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if len(approx) < 12 and area > img.shape[0] * img.shape[1] * AREA_THRESHOLD:
            contours_list.append(contour)

    return contours_list


def get_biggest_contour(contours_list):
    biggest_contour = None
    biggest_area = 0
    for contour in contours_list:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest_contour = contour
    return biggest_contour


def get_poly_hull(contour):
    # hull is the normal hull, poly hull is the polygon approximation
    hull = cv2.convexHull(contour)
    hull_poly = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
    return hull, hull_poly


def main():
    img = take_picture()
    show_img(img, "Original Image")
    edges = get_edges(img=img)
    show_img(edges, "Edges Image")
    contours = get_contours(edges)
    show_img(get_draw_contours(img, contours), "Contour Image")
    filtered_contours = get_relevant_contours(contours, img)
    biggest_contour = get_biggest_contour(filtered_contours)
    hull, poly_hull = get_poly_hull(biggest_contour)
    # drawing the hull to the img
    cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
    cv2.drawContours(img, [poly_hull], 0, (0, 255, 255), 2)
    show_img(img, "Output Image")


def get_results(img=None):
    if img is None:
        img = take_picture()
    edges = get_edges(img=img)
    contours = get_contours(edges)
    filtered_contours = get_relevant_contours(contours, img)
    biggest_contour = get_biggest_contour(filtered_contours)
    return get_poly_hull(biggest_contour)


# main()

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Capture frame-by-frame
# ret, frame = cap.read()
# # Our operations on the frame come here
# while np.sum(frame) == 0 or frame is None:
#     ret, frame = cap.read()
#     # Our operations on the frame come here
# img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

# tmp = cv2.resize(img, (1140, 640))
# cv2.imshow('Original Image', tmp)
# cv2.waitKey(0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(gray, 700, 900, apertureSize=5)
# edges = cv2.morphologyEx(edges, kernel=None, op=cv2.MORPH_CLOSE, iterations=2)
# edges = cv2.morphologyEx(edges, kernel=None, op=cv2.MORPH_OPEN, iterations=1)

# tmp = cv2.resize(edges, (1140, 640))
# cv2.imshow('edges Image', tmp)
# cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# tmp = img.copy()
# cv2.drawContours(tmp, contours, -1, (0, 255, 0), 2)
# tmp = cv2.resize(tmp, (1140, 640))
# cv2.imshow('Contour Image', tmp)
# cv2.waitKey(0)

# contours_list = []
#
# for contour in contours:
#     epsilon = 0.1 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
#     area = cv2.contourArea(contour)
#     if len(approx) < 12 and area > img.shape[0] * img.shape[1] * 0.1:
#         contours_list.append(contour)

# biggest_contour = None
# biggest_area = 0
# for contour in contours_list:
#     area = cv2.contourArea(contour)
#     if area > biggest_area:
#         biggest_area = area
#         biggest_contour = contour

# cv2.drawContours(img, [biggest_contour], -1, (120, 255, 31), 2)
# hull = cv2.convexHull(biggest_contour)
# hull_poly = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
# cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
# cv2.drawContours(img, [hull_poly], 0, (0, 255, 255), 2)

# tmp = cv2.resize(img, (1140, 640))
# cv2.imshow('Output Image', tmp)
# cv2.waitKey(0)
