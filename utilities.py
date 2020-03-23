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


def view_data_segments(xs, ys, y_final_plot, finalPowerSet, filename):
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
    plot = plt.subplot()
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    colourSegment = [0, 1, 2, 3, 4, 5]
    plt.set_cmap('Dark2')

    data = plot.scatter(xs, ys, c=colour)
    legend1 = plot.legend(*data.legend_elements(),
                        loc="upper right", title="Data segment", fontsize='small', title_fontsize='small')
    plot.add_artist(legend1)
    colourSegment = [0, 1, 2, 3, 4, 5]
    for l in range(num_segments):
        # print(l)
        start = l*20
        end = start + 20
        # print("end:", end)
        plt.set_cmap('Dark2')
        line = plot.plot(xs[start: end], y_final_plot[l].T, label='Regression Line:' +str(l))

    plot.legend(loc=2, fontsize='small', title_fontsize='small')
    # plt.savefig('Plots/'+filename[11:-4], dpi = 400)
    plt.show()
