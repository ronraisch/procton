import numpy as np
from Constants import *
from Board_detection_manager.Soldier_Detection import ImageProcessing


class Board:
    NUM_OF_PIECES_COLOR = 15

    def __init__(self):
        self.cells = np.zeros((NUMBER_OF_TRIANGLES))
        self.eaten = np.zeros((2))
        self.triangle_content = [[] for i in range(NUMBER_OF_TRIANGLES + 2)]
        self.eaten_content = [[] for i in range(2)]
        self.is_first_run = True

    def get_board_position(self):
        return self.cells

    def get_eaten(self):
        return self.eaten

    def get_board_content(self):
        return self.triangle_content

    def get_eaten_content(self):
        return self.eaten_content

    def set_board_positions(self, board_positions):
        self.cells = board_positions

    def set_board_content(self, board_content):
        self.triangle_content = board_content

    def update_from_camera(self, img, dice_result):
        white_circles, black_circles, result_img = ImageProcessing.get_detected_soldiers(img_input=img)
        self.camera2board(white_circles, black_circles, dice_result)

    def compare_distance_from_edge(self, piece):  # calcs distance piece from edge
        return min(np.abs(piece[1] - HEIGHT), np.abs(piece[1]))

    def locate_pieces(self, circles, color):
        """
        finding the piece's locations on the board (in which triangle are in) and
        updating it in cells and stacks_of_board - one of the colors
        :param circles: the circles found by hough transform
        :param color: color of the pieces
        """
        for circle in circles[0]:
            x_position = circle[0]
            y_position = circle[1]
            if x_position != 0:
                if (NUMBER_OF_TRIANGLES / 4 + 1) * A > x_position > NUMBER_OF_TRIANGLES / 4 * A:  # is eaten?
                    self.eaten[1 * (color == BLACK)] += 1
                    self.eaten_content[1 * (color == BLACK)].append(circle)
                else:
                    if x_position > NUMBER_OF_TRIANGLES / 4 * A:  # is after the middle?
                        x_position = x_position - DELTA
                    cell = int(x_position / A)
                    if y_position > Y_MIDDLE:  # is in the bottom half?
                        cell = int(NUMBER_OF_TRIANGLES - 1 - cell)
                    self.cells[cell] += color
                    self.triangle_content[cell + 1].append(circle)

    def camera2board(self, white_circles, black_circles, dice_result):  # takes the imageProcessing results and
        # converts it to a board state and the triangle content
        self.locate_pieces(white_circles, WHITE)
        self.locate_pieces(black_circles, BLACK)
        # to make it same with EMM
        self.set_board_positions([0.0]+list(np.flip(self.cells))+[0.0]+list(self.eaten)+dice_result)
        self.set_board_content(self.triangle_content[::-1] + self.eaten_content)
        if self.is_first_run:
            for stack in self.triangle_content:
                stack.sort(key=self.compare_distance_from_edge)
            self.is_first_run = False
