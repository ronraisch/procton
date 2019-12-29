import board_resize.perspective_change as pc
import Board_size_detection.board_detection as bd
import Dice_Detection.dice_recognition as dd
import soldier_detection_lights.soldier_detection as sd
from Board_size_detection.board_detection import take_picture, show_img
import numpy as np


def main():
    img = take_picture()

    soldiers_circles = sd.get_results(img)
    soldiers_img = sd.get_drew_circles(img, soldiers_circles)


    hull, poly_hull = bd.get_results(img)
    poly_hull = np.reshape(poly_hull, (4, 2))
    img = pc.four_point_transform(img, hull)
    # show_img(img, "perspective image")


    soldiers_img = pc.four_point_transform(soldiers_img, poly_hull)
    show_img(soldiers_img, "Soldiers Detection")

    dice_contours = dd.get_results(img)
    dice_img = dd.get_drew_contours(img,dice_contours)
    show_img(dice_img,"Dice Image")


main()
