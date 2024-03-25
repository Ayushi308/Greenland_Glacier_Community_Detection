import sys
import os
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

import community_detection

# Import data
data = np.load('30th_subsample.npy')

# Run community detection (we consider ONLY time series with less than p% missing data)
p = 0.20

# If you want a specific k, just specify it (e.g., k = 0.5)
# If k = None, we choose k as a high quantile of the distribution of all pairwise correlations
community_map, single_communities, signals = community_detection.infomap_communities(data,p,k=None)