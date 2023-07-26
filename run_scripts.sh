#!/bin/bash

read -p 'Enter the user whose configuration needs to be used for the script: ' user


# Run heatmap code
cd heatmap
python draw_heatmap.py $user


### Run scanpath visualizations
cd ..
cd scanpath_visualization
python ref_image_plotting.py $user
python video_plotting.py $user


### Run tagger
cd ..
cd tagging_tool
python feature_tagger_deluxe.py $user

### Run grouper

### Run analysis script

