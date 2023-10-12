#!/bin/sh

base_dest_dir="./31082023/./"
parameters= "S57" "S39" "S9" "S30" "S52" "S51" "S49" "S36" "S28" "S5" "S23" "S4" "S46" "S59" "S29" "S26" "S44" "S14"

# Create a temporary directory
temp_dir="./temp_squash_dir"
mkdir -p "$temp_dir"

# Loop over each parameter to copy the files from the directories into the temporary directory
for param in $parameters; do
    input_dir="./${param}"

    # Check if the directory exists before copying
    if [ -d "$input_dir" ]; then
        # Copy only the files from the directory to the temporary directory
        cp "$input_dir"/* "$temp_dir/"
        echo "Copied files from $input_dir into $temp_dir"
    else
        echo "Directory $input_dir does not exist. Skipping."
    fi
done

# Now squash the entire temporary directory into one .sqfs file
output_file="/home/ali_alouane/MA_BCI/squashfs_smr_data/pvc_c1.sqfs"
squash-dataset "$temp_dir" "$output_file"
echo "Squashed all files into $output_file"

# Clean up the temporary directory
rm -rf "$temp_dir"
