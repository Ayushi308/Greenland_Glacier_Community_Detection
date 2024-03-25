# -*- coding: utf-8 -*-
# importing libraries

import os
import numpy as np
import scipy.io
import numpy.ma as ma
from scipy import signal, spatial
from scipy.spatial.distance import pdist,cdist, squareform
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Network libraries
import networkx as nx
import infomap
import preprocessing
import pandas as pd

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Fabri Falasca
# fabrifalasca@gmail.com

'''
Given time series embedded in a spatiotemporal grid,
this code find clusters/modes based on their correlation
networks. The clustering problem is cast as a community detection
problem.
The community detection algorithm used is infomap
'''

# Function to compute the Adjacency matrix from the correlation matrix
def compute_adj_m(x,k):
    # inputs: - x is a correlation (distance) matrix
    #         - tau is a threshold
    adj_m = x.copy()
    # Step (1) set the diagonal of the correlation matrix to zero
    np.fill_diagonal(adj_m,0)
    # Step (2) for a given number i, if i >= tau ---> 1, if i < tau ---> 0
    adj_m[adj_m>=k]=1
    adj_m[adj_m<k]=0

    return adj_m

'''
Main function
Inputs

(a) dataset: numpy array. Shape: [time, Y dimension, X dimension].

(b) p: percentage of nan allowed in a time series. 
E.g., p = 0.1 means
that no more than 10% of the values in each time series can be nans

(c) k: minimum correlation needed for 2 time series to be connected.
E.g., k = 0.3, would mean that two time series are linked if their Pearson 
correlation is larger than 0.3
The default k is k = None. In this case we consider all possible pairwise 
correlations and set k as a high quantile (q = 0.95) of the distribution.
However, it can be specified by the user and set to whatever value in between 
k = 0 and k = 1
'''
def infomap_communities(data,p,k=None):

    data = preprocessing.remove_mean(data)
    data = preprocessing.remove_variance(data)

    # Dimension of the field
    dimX = np.shape(data)[2] # Points in X axis (long)
    dimY = np.shape(data)[1] # Points in Y axis (lat)
    dimT = np.shape(data)[0] # Points in time

    # Now, we consider ONLY time series with less than p% missing data

    check_ratios = []

    # Check: if a grid point is masked with nan at time t it should be masked at all t
    for i in range(dimY):
        for j in range(dimX):
            # number of nans
            n_nans = np.sum(np.isnan(data[:,i,j]))
            # ratio of nans
            ratio = n_nans/dimT
            check_ratios.append(ratio)
            if ratio > p: # if we have more nans than we want...
                data[:,i,j] = np.nan

    check_ratios = np.array(check_ratios) 
    # how many time series "made the cut"?
    n_ts = np.sum(check_ratios < p)/len(check_ratios) # percent
    print('Percentage of time series considered: '+str(np.round(n_ts*100,2))+' %')    

    # From a [dimT,dimY,dimX] array
    # to a list of (dimY x dimX) spectra of length dimV
    flat_data_masked = data.reshape(dimT,dimY*dimX).transpose()

    # indices where we have numbers (non-nans)
    # it will be useful later to assign a community label to each grid point
    #indices = np.where(~np.isnan(np.sum(flat_data_masked,axis=1)))[0]
    indices = []
    for i in range(len(flat_data_masked)):
    
        # number of nans
        n_nans = np.sum(np.isnan(flat_data_masked[i]))
        # if this is equal to the length of the time series
        # then this time series is all nan and it is useless to us
        # if not, we save the index
        if n_nans < dimT:
            indices.append(i)
        
    indices = np.array(indices)          

    flat_data = flat_data_masked[indices]

    df = pd.DataFrame(np.transpose(flat_data))
    rho = df.corr()
    corr_matrix = np.array(rho)


    if k == None:

        # Let's consider the histogram of all correlations
        # To do so we consider only the upper triangle 
        # of the correlation matrix
        upper_triangle = np.triu(corr_matrix)
        upper_triangle = upper_triangle[upper_triangle!=0]
        upper_triangle = upper_triangle[upper_triangle!=1]  
        k = np.quantile(upper_triangle,0.95)
    
    print('K parameter set to: '+str(k))

    # Compute Adjacency matrix
    adj_matrix = compute_adj_m(corr_matrix,k)

    # From an Adjacency matrix to a graph object
    graph = nx.Graph(adj_matrix)

    # Infomap
    print('Community detection via Infomap')

    im = infomap.Infomap("--two-level --verbose --silent")

    # Add linnks
    for e in graph.edges():
        im.addLink(*e)

    # run infomap
    im.run()

    print(f"Found {im.num_top_modules} communities")

    # List of nodes ids
    nodes = []
    # List of respective community
    communities = []

    for node, module in im.modules:

        nodes.append(node)
        communities.append(module)

    partition = np.transpose([nodes,communities])

    # How many communities?
    n_com = np.max(partition[:,1])

    # Compute an average signal from each community

    # Step (a)
    # In modes_indices we store the indices of nodes belonging to each community
    #  e.g., modes_indices[0] will contain all nodes in community 0 etc.
    modes_indices = []
    for i in np.arange(1,n_com+1):
        modes_indices.append(partition[partition[:,1]==i][:,0])
    #modes_indices = np.array(modes_indices)

    # Now compute the signals of each community as the cumulative anomaly inside
    signals = []
    for i in range(len(modes_indices)):
        # Extract the mode
        extract_mode = np.array(flat_data)[modes_indices[i]]
        # Compute the signal as the mean time series (nans are not considered)
        signal = np.nanmean(extract_mode,axis=0)
        # Store the result
        signals.append(signal)

    signals = np.array(signals)

    # Let's embed the results on the grid
    # for now this community maps is flattened
    community_grid_flattened = flat_data_masked[:,0] # at time t = 0
    community_grid_flattened[indices] = partition[:,1] # these are the labels
    community_map = community_grid_flattened.reshape(dimY,dimX)

    # Finally, let's also save single community maps
    # Output:
    # For each domain we have a 2-Dimensional map where grid points belonging to the domain have value 1
    # points that do not belong to the domain have are nan
    single_communities = []

    for i in range(n_com):
        i = i + 1 # labels start from 1 not from 0
        community = community_map.copy()
        community[community!=i]=np.nan
        community[community==i]=1
        single_communities.append(community)
    single_communities = np.array(single_communities)
    
    output_dir = 'output_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(signals)):
        fig = plt.figure(figsize=(15, 2))
        ax = fig.add_subplot(121)
    
        plt.imshow(np.flipud(single_communities[i]), cmap=plt.cm.prism)
        plt.title('Community #' + str(i), fontsize=19)
        file_name = os.path.join(output_dir, f'community_{i}_part.png')
        plt.savefig(file_name, bbox_inches='tight')
    
        plt.close()

    return community_map, single_communities, signals
