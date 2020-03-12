import sys

import numpy as np
from matplotlib import pyplot as plt
import utilities

#TODO:
#Basic_3 still gives a Cubic and I think it should be a quadratic
#Basic 4 segment 1 gives a Cubic and I think it should be a quadratic


segmentLength = 20
def splitSegment(xs, ys):
    divider = 0
    if(len(xs) >= segmentLength):
        divider = len(xs)/ segmentLength

    return np.split(xs, divider), np.split(ys, divider)

def leastSquares(sampleX, sampleY, poly):
    #Calculating coefficients for target polynomial
    xs = np.array(sampleX)
    ys = np.array(sampleY)
    powColumns = np.ones((len(xs), 1))
    xArray = np.zeros((len(xs), 1))
    #Check for horizontal line
    if(poly > 1):
        #Matrix Multiplicaition for Least Square Regression to give 'a' (coefficients)
        for p in range(1, poly):
            for i in range(len(xs)):
                xValue = np.power(xs[i], p)
                xArray[i, 0] = xValue
            powColumns = np.concatenate((powColumns, xArray), axis=1)
    w = np.linalg.inv(np.dot(powColumns.T, powColumns))
    a = np.dot(np.dot(w, powColumns.T), ys)

    #Make y_plot of regression
    y_plot = np.array([])
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

def squaredError(y_plot, ys, a):
    error = 0.
    for i in range((len(ys))):
        diff = y_plot[i] - ys[i]
        squaredDiff = pow(diff, 2)
        error = error + squaredDiff

    return error, y_plot, a

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
            print("Enter '--plot' as second argument in order to view the graph")
            return plot, fileName
    else:
        print("Enter a valid file format: filename.format")
        exit()


def main():

    plot, fileName = inputOutput()

    xs, ys = utilities.load_points_from_file(fileName)
    sampleX, sampleY = splitSegment(xs, ys)
    limit = 3



    ySetArray = []
    totalError = 0.
    bestYs = []
    powerSet = []
    bestPowerSet = []
    finalPowerSet = []
    for l in range(len(sampleX)):
        errorArray = np.array([])
        actualErrorArray = np.array([])
        bestError = 0.
        bestErrorIndex = 0
        for p in range(1, limit+1):
            y_plot, a = leastSquares(sampleX[l], sampleY[l], p)
            error, y_set, a = squaredError(y_plot, ys[l*segmentLength: l*segmentLength+segmentLength], a)
            actualErrorArray = np.append(actualErrorArray, error)
            #Scaled Error
            error = error * p
            errorArray = np.append(errorArray, error)
            ySetArray.append(y_set)
            powerSet.append(a)


        #Sine function
        y_plot, a = leastSquaresSin(sampleX[l], sampleY[l])
        sinePlot = np.array(y_plot)
        error, y_set, a = squaredError(y_plot, ys[l*segmentLength: l*segmentLength+segmentLength], a)
        actualErrorArray = np.append(actualErrorArray, error)
        error = error * 1.25
        errorArray = np.append(errorArray, error)
        ySetArray.append(y_set)
        powerSet.append(a)
        y_plot = None
        a = None
        error = 0.
        y_set = None

        bestErrorIndex = np.argmin(errorArray)
        bestError = actualErrorArray[bestErrorIndex]

        actualError = actualErrorArray[bestErrorIndex]
        bestPowerSet = powerSet[bestErrorIndex]
        finalPowerSet.append(bestErrorIndex)
        bestYs.append(ySetArray[bestErrorIndex])

        totalError = totalError + actualError

        print("Best Error from coefficients:", bestPowerSet, "with error:", bestError, "for segment", l)
        powerSet.clear()
        ySetArray.clear()
    y_final_plot = np.array(bestYs)

    print("TotalError:", totalError)
    if(plot == True):
        utilities.view_data_segments(xs , ys, y_final_plot, finalPowerSet)

main()