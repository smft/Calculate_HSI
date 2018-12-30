#!/usr/bin/env python

import glob
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

""" test!!!test """
def calculate_ndvi_max(ndvi,light):
    light_max=np.max(light)
    olsnor=(light)/(light_max)
    hsi=(1-ndvi)+olsnor
    hsi=hsi/((1-olsnor)+ndvi+ndvi*olsnor)
    return hsi

###################
""" test!!!test """
###################
ndvi=pickle.load(open('NDVI/max_ndvi.pickle'))
light=pickle.load(open('LIGHT/night_light.pickle'))
hsi=calculate_ndvi_max(ndvi,light)
urban_type=np.zeros_like(hsi)
for i,cell in enumerate(hsi):
    for j,idx in enumerate(cell):
        if idx<55:
            urban_type[i,j]=0
        elif 55<=idx<85:
            urban_type[i,j]=31
        elif 85<=idx<95:
            urban_type[i,j]=32
        else
            urban_type[i,j]=33

plt.subplot(2,2,1)
plt.imshow(ndvi)
plt.colorbar()
plt.title('NDVI')

plt.subplot(2,2,2)
plt.imshow(light)
plt.colorbar()
plt.title('LIGHT')

plt.subplot(2,2,3)
plt.imshow(hsi)
plt.colorbar()
plt.title('HSI')

plt.subplot(2,2,4)
plt.imshow(urban_type)
plt.colorbar()
plt.title('URBAN')

plt.show()
