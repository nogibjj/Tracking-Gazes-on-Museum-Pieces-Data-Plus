# Tracking-Gazes-on-Museum-Pieces-Data-Plus
This is a data plus project where we track the gazes of participants who were looking at a specific Truscan art piece.

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

