import sys
import pandas as pd
import scipy
import matplotlib
import sklearn
import numpy
import csv

def readLabels(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0
    labels = []

    for row in reader:
        labels.append(row)
        rownum += 1

    ifile.close()
    labels = numpy.delete(labels, 0, 1)
    labels = numpy.delete(labels, 0, 1)
    labels = numpy.delete(labels, 0, 1)
    labels = numpy.delete(labels, 0, 1)
    return labels

def readcsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0
    data = []

    for row in reader:
        data.append(row)
        rownum += 1

    ifile.close()
    return data