import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, y_final_plot):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20

    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    colourSegment = ['r-', 'g-', 'b-', 'y-']
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    for l in range(num_segments+1):
        #print(y_final_plot.shape)
        print("start:", l*20)
        print("end:", l * 20 + 20)
        plt.plot(xs[l*20: (l*20)+20], y_final_plot[l].T, colourSegment[l])
    plt.show()
