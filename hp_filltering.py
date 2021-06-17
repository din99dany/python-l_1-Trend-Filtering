#/usr/bin/python3 
import math
import sys
import numpy as np
import scipy as scipy
import scipy.optimize as opt
import matplotlib.pyplot as plt


landa = 10000

# Load time series data: S&P 500 price log.
dataset = np.loadtxt(open('snp500.txt', 'rb'), delimiter=",", skiprows=1)  # y
datasetLength = dataset.size

D = np.zeros( ( datasetLength - 2, datasetLength ) )
for i in range( datasetLength - 2) :
    D[ i ][ i ] = 1 
    D[ i ][ i + 1 ] = -2 
    D[ i ][ i + 2 ] = 1 
    
toInvers = np.identity( datasetLength ) + (2*landa)*( np.transpose(D) @ D )    
inversed = np.linalg.inv( toInvers )


xhp  = inversed @ dataset 

plt.plot( np.linspace( 1, datasetLength, datasetLength ), dataset, label="data" )
plt.plot( np.linspace( 1, datasetLength, datasetLength ), xhp, label="xhp" )
plt.legend()
plt.show()  