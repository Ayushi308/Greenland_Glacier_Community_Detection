import sys
import os
import time
import numpy as np

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def remove_mean(climate_field):
	"""
	Normalizes time series to zero mean
	Input:
		climate_field: Numpy mask array. The climate field time series
	Returns:
		climate_filed: Numpy mask array. The climate field zero mean time series
	"""
	mean = np.nanmean(climate_field,axis=0);
	climate_field = climate_field - mean;
	return climate_field;



def remove_variance(climate_field):
    """ 
    Normalizes time series to unit variance
    Input:
        climate_field: Numpy mask array. The climate field time series
    Returns:
        climate_filed: Numpy mask array. The climate field unit variance time series
    """
    std = np.nanstd(climate_field,axis = 0);
    climate_field = climate_field/std;
    return climate_field;
