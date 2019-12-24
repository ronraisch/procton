path = "C:\\Users\\t8709339\\Desktop\\talpiot\\procton\\Board_detection_manager\\Soldier_Detection\\stds"
import os


def frame_movie(filename):
    import cv2
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(filename + " " + "frame%d.png" % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


for filename in os.listdir(path):
    if filename.endswith(".mp4"):
        frame_movie(path + "\\" + filename)
        continue
    else:
        continue
