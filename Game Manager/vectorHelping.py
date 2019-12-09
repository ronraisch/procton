import numpy as np


def vector_length(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)


def vector_length_edge_points(vec1, vec2):
    return np.sqrt((vec2[0] - vec1[0])**2 + (vec2[1] - vec1[0])**2)


def get_unit_vector(vec):
    return vec / vector_length(vec)
