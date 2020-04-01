import sys
import pandas as pd
import scipy
import matplotlib
import sklearn
import numpy
import csv

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