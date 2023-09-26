#!/bin/sh

base_dest_dir="./31082023"
parameters="S5 S9 S20 S2 S19 S14"

# Create a temporary directory
temp_dir="./temp_squash_dir"
mkdir -p "$temp_dir"

# Loop over each parameter to copy the directories into the temporary directory
for param in $parameters; do
    input_dir="./${param}"

    # Check if the directory exists before copying
    if [ -d "$input_dir" ]; then
        cp -r "$input_dir" "$temp_dir/"
        echo "Copied $input_dir into $temp_dir"
    else
        echo "Directory $input_dir does not exist. Skipping."
    fi
done

# Now squash the entire temporary directory into one .sqfs file
output_file="/home/ali_alouane/MA_BCI/squashfs_smr_data/hpo_best_pvc.sqfs"
squash-dataset "$temp_dir" "$output_file"
echo "Squashed all directories into $output_file"

# Clean up the temporary directory
rm -rf "$temp_dir"