import sys

import numpy as np
from matplotlib import pyplot as plt
import utilities
#Define Segment Length
segmentLength = 20

#Split input arrays to segments
def splitSegment(xs, ys):
    divider = 0
    if(len(xs) >= segmentLength):
        divider = len(xs)/ segmentLength

    return np.split(xs, divider), np.split(ys, divider)

def leastSquaresPoly(sampleX, sampleY, poly):
    #Calculating coefficients for target polynomial
    xs = np.array(sampleX)
    ys = np.array(sampleY)
    powColumns = np.ones((len(xs), 1))
    xArray = np.zeros((len(xs), 1))
    if(poly > 1):                                                       # Check for horizontal line
        for p in range(1, poly):                                        # Matrix Multiplicaition for Least Square
            for i in range(len(xs)):                                    #Regression to give 'a' (coefficients)
                xValue = np.power(xs[i], p)
                xArray[i, 0] = xValue
            powColumns = np.concatenate((powColumns, xArray), axis=1)   #Creating 'X' matrix with x powers from dataset
    w = np.linalg.inv(np.dot(powColumns.T, powColumns))
    a = np.dot(np.dot(w, powColumns.T), ys)
    y_plot = np.array([])                                               #Make y_plot of regression
    y_val = 0.
    for x in range(len(xs)):
        for i in range(len(a)):
            y_val = y_val + a[i] * pow(xs[x], i)
        y_plot = np.append(y_plot, y_val)
        y_val = 0.

    return y_plot, a

def leastSquaresSin(sampleX, sampleY):

    sinX = np.sin(sampleX)
    ones = np.array([np.ones(len(sampleX))])
    xi = np.column_stack((ones.T, sinX))
    w = np.linalg.inv(np.dot(xi.T, xi))
    a = np.dot(np.dot(w, xi.T), sampleY)

    #Make y_plot of regression
    y_plot = np.array([])
    y_val = 0.

    for x in range(len(sampleX)):
        y_val = a[0] + a[1]*sinX[x]
        y_plot = np.append(y_plot, y_val)

    return y_plot, a

#LSR for Exponential Fucntions
def leastSquaresExp(sampleX, sampleY):

    sinX = np.exp(sampleX)
    ones = np.array([np.ones(len(sampleX))])
    xi = np.column_stack((ones.T, sinX))
    w = np.linalg.inv(np.dot(xi.T, xi))
    a = np.dot(np.dot(w, xi.T), sampleY)

    #Make y_plot of regression
    y_plot = np.array([])
    y_val = 0.

    for x in range(len(sampleX)):
        y_val = a[0] + a[1]*sinX[x]
        y_plot = np.append(y_plot, y_val)

    return y_plot, a

#Func to calculate Squared Error
def squaredError(y_plot, ys, a):
    error = 0.
    for i in range((len(ys))):
        diff = y_plot[i] - ys[i]
        squaredDiff = pow(diff, 2)
        error = error + squaredDiff

    return error, y_plot, a

#I/O function to read in arguments from commandline
def inputOutput():
    if (len(sys.argv) > 1):
        fileName = "train_data/" + sys.argv[1]
        if(len(sys.argv) == 3):
            print(sys.argv[2])
            if(sys.argv[2] == '--plot'):
                plot = True
                return plot, fileName
        else:
            plot = False
            return plot, fileName
    else:
        print("Enter a valid file format: filename.format")
        exit()

#Altered view_data_segment function from utilities to give coloured plots and legends
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

def main():

    plot, fileName = inputOutput()

    xs, ys = utilities.load_points_from_file(fileName)
    sampleX, sampleY = splitSegment(xs, ys)
    limit = 4



    ySetArray = []
    totalError = 0.
    bestYs = []
    powerSet = []
    bestPowerSet = []
    finalPowerSet = []
    sampl = np.random.uniform(low=10000, high=1000000, size=(20,1))
    for l in range(len(sampleX)):
        min = np.min(sampleY[l])
        max = np.max(sampleY[l])

        ScaledErrorArray = np.array([])
        actualErrorArray = np.array([])
        bestError = 0.
        bestErrorIndex = 0
        for p in range(1, limit+1):
            #Make Quadratic functions unreachable with errors always too high (max float value)
            if(p == 3):
                y_plot = sampl
                a = [0., 0., 0.]
                error = sys.float_info.max
                y_set = sampl
                actualErrorArray = np.append(actualErrorArray, error)
                error = error
                ScaledErrorArray = np.append(ScaledErrorArray, error)
                ySetArray.append(y_set)
                powerSet.append(a)
            else:
            #Regular LSR plot that only works for Linear and Cubic
                y_plot, a = leastSquaresPoly(sampleX[l], sampleY[l], p)
                error, y_set, a = squaredError(y_plot, ys[l*segmentLength: l*segmentLength+segmentLength], a)
                #Actual Error value
                actualErrorArray = np.append(actualErrorArray, error)
                #Scaled Error value
                error = error * (p)
                ScaledErrorArray = np.append(ScaledErrorArray, error)
                ySetArray.append(y_set)
                powerSet.append(a)
        #Sine function
        y_plot, a = leastSquaresSin(sampleX[l], sampleY[l])
        sinePlot = np.array(y_plot)
        error, y_set, a = squaredError(y_plot, ys[l*segmentLength: l*segmentLength+segmentLength], a)
        actualErrorArray = np.append(actualErrorArray, error)
        error = error * 2
        ScaledErrorArray = np.append(ScaledErrorArray, error)
        ySetArray.append(y_set)
        powerSet.append(a)

        y_plot = None
        a = None
        error = 0.
        y_set = None
        #Select Best Error from Scaled errors
        bestErrorIndex = np.argmin(ScaledErrorArray)

        bestError = actualErrorArray[bestErrorIndex]
        #Select actual error using index from scaled error
        actualError = actualErrorArray[bestErrorIndex]
        bestPowerSet = powerSet[bestErrorIndex]
        finalPowerSet.append(bestErrorIndex)
        bestYs.append(ySetArray[bestErrorIndex])
        totalError = totalError + actualError
        powerSet.clear()
        ySetArray.clear()

    y_final_plot = np.array(bestYs)
    print(totalError)
    if(plot == True):
        view_data_segments(xs , ys, y_final_plot, finalPowerSet, fileName)

main()