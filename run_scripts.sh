#!/bin/bash


read -p 'Enter the user whose configuration needs to be used for the script: ' user
# Change to the directory containing the Python script
cd heatmap
# python draw_heatmap.py $user

cd ..
cd scanpath_visualization
python ref_image_plotting.py $user
python video_plotting.py $user

cd ..
cd tagging_tool
python feature_tagger_deluxe.py

