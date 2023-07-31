#!/bin/bash

pip install -r requirements.txt
read -p 'Enter the user whose configuration needs to be used for the script: ' user

# Create the reference image and tag it
cd heatmap
python reference_image_creator.py $user
cd ../tagging_tool
python feature_tagger.py $user

# Create heatmaps and map the data from video to reference image
cd ../heatmap
python draw_heatmap.py $user

# Run the Scanpath Visualisations
cd ../scanpath_visualization
python ref_image_plotting.py $user
python video_plotting.py $user

### Run tag
cd ../tagging_tool
python gaze_tag_mapper.py $user

# Run grouper
python grouping_csvs.py $user

# Run analysis script
cd ../analysis
python analysis.py $user