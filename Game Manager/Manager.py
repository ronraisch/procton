import numpy as np

game_over = False
we_start = True
pieces_left = 30
pixel_to_real_ratio = None
board_corners = float[4][2]
mid_width = None
cell_parameters = float[24][4] #(Xright, Xleft, Yup, Ydown)
from board detection import BoardDetection
import vectorHelping

# WE WORK IN CENTIMETERS!!!

board_length = 49
board_width = 44
board_mid_length= 3
board_side_length = 2

def initial_scan():
    """
    Updates the manager according to the board's properties
    :return: Nothing
    """
    board_corners = BoardDetection()
    # get_board_parameters()


def get_board_parameters():
    x_full_vector = board_corners[1] - board_corners[0]

    y_full_vector = (board_corners[0][1] - board_corners[1][1] + board_corners[3][1] - board_corners[2][1])/4
    for i in range(24):
        if -1 < i < 6 or 11 < i < 18:
            cell_parameters[i][0] = board_corners[3][0] + ((i % 12) + 1) * x_full_vector
            cell_parameters[i][1] = board_corners[3][0] + (i % 12) * x_full_vector
        else:
            cell_parameters[i][0] = board_corners[0][0] - ((12 - i) % 12 - 1) * x_full_vector
            cell_parameters[i][1] = board_corners[0][0] - ((12 - i) % 12) * x_full_vector
        if i/12 < 1:
            cell_parameters[i][2] = board_corners[0][1]
            cell_parameters[i][3] = board_corners[0][1] - y_full_vector
        else:
            cell_parameters[i][2] = board_corners[2][1] + y_full_vector
            cell_parameters[i][3] = board_corners[2][1]


def roll_dice():
    """
    rolling the dices
    :return: whether succeeded
    """
    return 1


def get_locations():
    """
    gets the piece's locations and dice roll from the image processing module
    :return: a list(size 30) of the piece's locations, tuple of the dice rolls
    """
    return 2


def check_locations(locations_vector):
    """
    checks if the current board state is fine
    :param locations_vector: locations of the pieces
    :return: integer representing the problem after fixing it
    """
    return 3


def translate_locations_to_board(location_vector):
    """
    translates from picture coordinates to backgammon-board vector
    :param location_vector: the pieces locations
    :return: a vector describing the current board
    """
    return 5


def get_move(board_state, dice_roll):
    """
    gets our next move from the algorithm
    :param board_state: the board-vector
    :param dice_roll: the dice roll
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
    roll_dice()
    location_vector, dice_roll = get_locations()
    found_error = check_locations(location_vector)
    while found_error:  # and handle problems
        found_error = check_locations(location_vector)
    board_state = translate_locations_to_board(location_vector)
    move = get_move(board_state, dice_roll)
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
