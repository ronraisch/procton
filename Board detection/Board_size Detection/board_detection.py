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


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.001,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

tmp = cv2.resize(img, (1140, 640))
cv2.imshow('output Image', tmp)
cv2.waitKey(0)

#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# # gray = gray.astype(np.float)
# # tmp = cv2.resize(gray, (1140, 640))
# # cv2.imshow('gray Image', tmp)
# # cv2.waitKey(0)
# dst = cv2.goodFeaturesToTrack(gray,maxCorners=6,qualityLevel=3,minDistance=700)
# tmp = cv2.resize(dst, (1140, 640))
# cv2.imshow('corner Image', tmp)
# cv2.waitKey(0)
#
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(src=dst,kernel=None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# tmp = cv2.resize(img, (1140, 640))
#
# cv2.imshow('dst',tmp)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()