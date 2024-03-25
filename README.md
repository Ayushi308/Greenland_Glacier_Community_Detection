# Greenland_Glacier_Community_Detection
Community Detection Algorithm Over Greenland Glaciers (code located inside Github_Code Folder)


# Preprocessing_Dataset
This folder includes two notebooks one for the entire Greenland Glacier region and one for the southern region of Greenland Glaciers. The notebooks are designed to take a folder of images seperated by years and build a 3D numpy array and store it as a npy file to send into the community detection algorithm.

# Tutorial Notebooks
For the code in the community_detection and community_detection_script2.py, this folder contains a detailed notebook walking through the code for these algorithms

# Community_Detection_script2.py
This script is the only place where you need to enter you npy file and p percentage value (for missing values). Running this script will run you npy dataset through the comunity detection algorithm. The batch script for this is also included under batch_script.txt
