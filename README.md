# Generalized Framework for Gaze Tracking
This repo is part of the work done by the team 19: Neurocities and Ruinscapes as part of Data+ at Duke University. We (Aditya, April and Eric) present a generalized framework that can be applied to any gaze tracking dataset that contains a gaze video along with an accompanying CSV of gaze points. 

We have utilized a wide array of image processing and statistical techniques to create the entire framework. 

# Framework Overview
![Alt text](Flowchart.jpg "Title")

# Steps to run the framework: 

1. Data Preparation (Sample data in the required format provided in this repo)
- Create a data folder in your root path. 
- Create a folder for the specific experiment you are organising. Ensure that this name matches what is present in the art piece in the config.py file. 
- Place all the separate participants within this specific folder. 
- If you have a reference image you wish to plot the heatmap on, place the same in the same folder. 

2. Running all the scripts. 
- Changes to config.py file 
    - Add a new user for yourself. 
    - To use the default framework, inherit variables from the base class (which if reuqired can be overwritten). 

- To run the entire framework: bash run_scripts.sh $user
- To run individual scripts (the same patterns follow for all)
    - cd heatmap [or the other functions]
    - python main.py $user

3. Framework Overview

