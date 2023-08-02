#!/bin/bash

pip install -r requirements.txt
read -p 'Enter the user whose configuration needs to be used for the script: ' user

# Create the reference image and tag it
cd create_reference_image
python main.py $user
cd ../tagging_tool
python main.py $user

# Create heatmaps and map the data from video to reference image
cd ../heatmap
python main.py $user

# Run the Scanpath Visualisations
cd ../scanpath_visualization
python main.py $user

### Run mapper and grouper script
cd ../mapping_tool
python main.py $user

# Run analysis script
cd ../analysis
python main.py $user