import numpy as np

game_over = False
we_start = True
pieces_left = 30
board_corners = float[4][2]
board_origin = np.array(2)
x_vector_unit = np.array(2)
y_vector_unit = np.array(2)
cell_parameters = float[24][4][2] # 24 segments, 4 corners each
real_life_to_camera_ratio = 0
IN, RIGHT, LEFT, UP, DOWN = 0, 1, -1, 2, -2
from board detection import BoardDetection
import vectorHelping

# WE WORK IN CENTIMETERS!!!

board_length = 49
board_width = 44
board_mid_length= 3
board_side_length = 2
board_x_no_distraction_length = board_length - board_mid_length - 2*board_side_length
board_y_no_distraction_length = board_width - 2*board_side_length

def initial_scan():
    """
    Updates the manager according to the board's properties
    :return: Nothing
    """
    board_corners = BoardDetection()
    real_life_to_camera_ratio = vectorHelping.vector_length_edge_points(board_origin[0], board_origin[1]/49)
    # get_board_parameters()


def get_board_parameters():
    board_origin = board_corners[3]
    x_vector = board_corners[1] - board_corners[0]
    x_vector_unit = vectorHelping.get_unit_vector(x_vector) * real_life_to_camera_ratio
    y_full_vector = - board_corners[0] + board_corners[3]
    y_vector_unit = vectorHelping.get_unit_vector(y_full_vector) * real_life_to_camera_ratio
    x_segment_part = x_vector_unit * board_x_no_distraction_length / 12
    y_segment_part = y_vector_unit * board_y_no_distraction_length / 2
    cell_parameters[0][0] = board_origin + x_vector_unit * 0 + 2 * y_vector_unit
    cell_parameters[0][1] = cell_parameters[0][0] + x_segment_part
    cell_parameters[0][2] = cell_parameters[0][1] - y_segment_part
    cell_parameters[0][3] = cell_parameters[0][0] - y_segment_part
    for i in range(1, 6):
        cell_parameters[i][0] = cell_parameters[i-1][1]
        cell_parameters[i][1] = cell_parameters[i][0] + x_segment_part
        cell_parameters[i][3] = cell_parameters[i-1][2]
        cell_parameters[i][2] = cell_parameters[i][3] + x_segment_part
    for i in range(23, 17, -1):
        cell_parameters[i][0] = cell_parameters[23-i][3]
        cell_parameters[i][1] = cell_parameters[23-i][2]
        cell_parameters[i][2] = cell_parameters[i][1] - y_segment_part
        cell_parameters[i][3] = cell_parameters[i][0] - y_segment_part
    cell_parameters[6][0] = cell_parameters[5][1] + x_vector_unit*(board_mid_length * real_life_to_camera_ratio)
    cell_parameters[6][1] = cell_parameters[5][2] + x_vector_unit*(board_mid_length * real_life_to_camera_ratio)
    cell_parameters[6][2]= cell_parameters[6][1] - y_segment_part
    cell_parameters[6][3] = cell_parameters[6][3] - y_segment_part
    for i in range(7,12):
        cell_parameters[i][0] = cell_parameters[i - 1][1]
        cell_parameters[i][1] = cell_parameters[i][0] + x_segment_part
        cell_parameters[i][3] = cell_parameters[i - 1][2]
        cell_parameters[i][2] = cell_parameters[i][3] + x_segment_part
    for i in range(17, 11, -1):
        cell_parameters[i][0] = cell_parameters[23-i][3]
        cell_parameters[i][1] = cell_parameters[23-i][2]
        cell_parameters[i][2] = cell_parameters[i][1] - y_segment_part
        cell_parameters[i][3] = cell_parameters[i][0] - y_segment_part


def calc_triangle_area(point1, point2, point3):
    return np.abs(point1[0]*(point2[1] - point3[1]) + point2[0]*(point3[1] - point2[1] +
                                                                 point3[0]*(point1[1] - point2[1])))/2

def is_between_4_corners(corners, point):
    """
    :param corners: shape(4,2) np array of 4 points
    :param point: a points
    :return: whether inside the rectangle defined by the corners (0 = in, 1 = right, -1 = left,
    2 = up, -2 = down)
    """
    req_area = vectorHelping.vector_length_edge_points(corners[0], corners[1]) * \
               vectorHelping.vector_length_edge_points(corners[2], corners[3])
    trig_area1 = calc_triangle_area(corners[0], corners[1], point)
    trig_area2 = calc_triangle_area(corners[1], corners[2], point)
    trig_area3 = calc_triangle_area(corners[2], corners[3], point)
    trig_area4 = calc_triangle_area(corners[3], corners[0], point)
    if np.abs(req_area - trig_area1 - trig_area2 - trig_area3 - trig_area4) < 0.05:
        return True
    return False


def roll_dice():
    """
    rolling the dices
    :return: whether succeeded
    """
    return 1


def get_locations(new_photo_filename, fake_dice_roll):
    """
    gets the piece's locations and dice roll from the image processing module
    :return: 2 numpy arrays (size 15,2) of the piece's locations.
    """
    white_coordinates = np.ndarray((15, 2))
    black_coordinates = np.ndarray((15, 2))
    [white_circles, black_circles] = ImageProcessing.get_detected_soldiers(new_photo_filename)
    counter = 0
    for circle in white_circles:
        if circle is None:
            break
        white_coordinates[counter][0], white_coordinates[counter][1] = circle[0], circle[1]
        counter += 1
    counter = 0
    for circle in black_circles:
        if circle is None:
            break
        black_coordinates[counter][0], black_coordinates[counter][1] = circle[0], circle[1]
        counter += 1

    return white_coordinates, black_coordinates, fake_dice_roll


def check_locations(locations_vector):
    """
    checks if the current board state is fine
    :param locations_vector: locations of the pieces
    :return: integer representing the problem after fixing it
    """
    return 3


def translate_point_to_board(point):
    theta = np.arctan(x_vector_unit[1] / x_vector_unit[0])
    point[0], point[1] = point[0]*np.cos(theta) - point[1]*np.sin(theta), point[0]*np.sin(theta) + point[1]*np.cos(theta)
    return point

def translate_locations_to_board(white_coordinates, black_coordinates, dice_roll):
    """
       translates from picture coordinates to backgammon-board vector
       :param location_vector: the pieces locations
       :return: a vector describing the current board
    """
    board_vector = np.array(30)
    board_vector[28] = dice_roll[0]
    board_vector[29] = dice_roll[1]
    board_vector[0] = 0  # winners white
    board_vector[25] = 0  # winners black
    board_vector[26] = 0  # eaten white
    board_vector[27] = 0  # eaten black
    for i in range(len(white_coordinates)):
        if white_coordinates[i] is None:
            break
        translated_coordinates = translate_point_to_board(white_coordinates[i])
        for j in range(24):
            if is_between_4_corners(cell_parameters[j],translated_coordinates):
                if board_vector[24-j] < -1:
                    print("Black and white in place "+ str(24-j))
                    return
                board_vector[24-j] += 1
                break
        print("Didnt find place for a piece")
        return
    for i in range(len(black_coordinates)):
        if black_coordinates[i] is None:
            break
        translated_coordinates = translate_point_to_board(black_coordinates[i])
        for j in range(24):
            if is_between_4_corners(cell_parameters[j], translated_coordinates):
                if board_vector[24-j] > 1:
                    print("Black and white in place "+ str(24-j))
                    return
                board_vector[24 - j] -= 1
                break
        print("Didnt find place for a black piece")
        return
    return board_vector


def get_move(board_state):
    """
    gets our next move from the algorithm
    :param board_state: the board-vector
    :return: the decided move
    """
    return 6


def translate_move_to_xy(move):
    """
    translate the decided move to real-world x,y starting and finishing coordinates
    :param move: the decided move
    :return: the start and end xs and ys
    """
    return 7


def do_turn(command):
    """
    do the turn mechanically
    :param command: the xes and yes to start and end for all pieces
    :return: nothing
    """
    return 8


def check_if_board_valid(location_vector):
    """
    Checking whether the new board we made is valid and fixing it
    :param location_vector: the locations of the pieces
    :return: nothing
    """
    return 9


def opponents_turn():
    """
    waiting for the opponent to make their move (or seeing it)
    :return: nothing (ends at opponent's turn end)
    """
    return 11


def our_turn():
    """
    Doing our turn
    :return:
    """
    dice_roll = roll_dice()
    white_location_vector, black_location_vector,  dice_roll = get_locations(dice_roll)
    # found_error = check_locations(location_vector)
    # while found_error:  # and handle problems
    #     found_error = check_locations(location_vector)
    board_state = translate_locations_to_board(white_location_vector, black_location_vector, dice_roll)
    move = get_move(board_state)
    command = translate_move_to_xy(move)
    do_turn(command)
    location_vector = get_locations()
    check_if_board_valid(location_vector)
    return 12


def winner():
    """
    The end of the game - showing the winner and stuff
    :return: nothing
    """
    return 13


def main():
    __init__()
    while not game_over:
        if we_start:
            our_turn()
            we_start = not we_start
        else:
            opponents_turn()
            we_start = not we_start
    winner()
