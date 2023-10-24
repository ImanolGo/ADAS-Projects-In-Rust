#!/bin/bash

# Create the main directory
mkdir ADAS-Projects-In-Rust
cd ADAS-Projects-In-Rust

# Initialize git
git init

# Create .gitignore and LICENSE
touch .gitignore
touch LICENSE

# Create the scripts folder
mkdir scripts
touch scripts/.gitkeep

# Create activity folders
activities=("Activity1-Lane-Detection"
            "Activity2-Traffic-Sign-Recognition"
            "Activity3-Speed-Limit-Recognition"
            "Activity4-Object-Detection-YOLOv3"
            "Activity5-Street-Light-Detection"
            "Activity6-Instrument-Cluster"
            "Activity7-Integration")

for activity in "${activities[@]}"; do
    mkdir $activity
    cd $activity
    touch README.md
    mkdir src
    mkdir data
    mkdir docs
    mkdir tests
    touch src/.gitkeep
    touch data/.gitkeep
    touch docs/.gitkeep
    touch tests/.gitkeep
    cd ..
done

# Create root README
echo "# ADAS Projects In Rust" > README.md

# Add everything to git and commit
git add .
git commit -m "Initial commit with directory structure."
