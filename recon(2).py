import sys
import os
import time
import numpy as np
from leapctype import *
from crip.io import imreadTiff,imwriteTiff
import os
import glob
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib
from leap_preprocessing_algorithms import *


numCols = 220
numAngles = 180
pixelSize = 0.125
numRows = 220
leapct = tomographicModels()

leapct.set_conebeam(180, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1)-(0.37)/0.125, \
                    leapct.setAngleArray(numAngles, (360-360/numAngles)), 213, 233)
leapct.set_volume(256, 256, 256, voxelWidth=0.1, voxelHeight=0.1, offsetX=None, offsetY=None, offsetZ=None)
leapct.set_rampFilter(12)
postlog_data = imreadTiff(rf'ProjPred.tif')

N_H1 = leapct.optimalFFTsize(2 * numRows)
N_H2 = leapct.optimalFFTsize(2 * numCols)
y = (np.array(range(N_H1), dtype=np.float32) - N_H1 // 2)
y = y / np.max(y)
x = (np.array(range(N_H2), dtype=np.float32) - N_H2 // 2)
x = x / np.max(x)
y, x = np.meshgrid(y, x, indexing='ij')

c = 9.0
b = (c + 1.0 - 0.5 * (1.0 + c)) / c

H = (1.0 - b) + b / (1.0 + c * np.sqrt(x ** 2 + y ** 2))
H = np.fft.fftshift(H)

g = leapct.allocateProjections()
f = leapct.allocateVolume()
g[:] = postlog_data
detectorDeblur_RichardsonLucy(leapct, g, H, numIter=10, isAttenuationData=True)
outlierCorrection(leapct,g)
ringRemoval(leapct, g, 1.0-0.99, 1.0e2, 20, 0.05)
leapct.FBP(g,f)
recon = np.array(f,dtype=np.float32)
imwriteTiff(recon, rf'low_dose_70ua_recon_my.tif')