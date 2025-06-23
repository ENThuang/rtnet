#!/bin/bash

# Define the source and destination directories
source_dir="/nas/yirui.wang/exps/Fudan_HN_LN_paper/SwinBackbone_fp16"
destination_dir="/home/yirui.wang/Desktop/HN_LN_paper_res_swinfp16_epoch100"

# Loop through the subdirectories RUN1 to RUN4
for run in RUN0 RUN1
do
    for fold in fold0 fold1 fold2 fold3 fold4
    do
        # Construct the source file path
        source_file="${source_dir}/${run}/${fold}/epoch_100_fuseall/swin_25d_fudan_hn_ln_bce_loss_dual_maxpool_ENE/predictions.txt"

        # Check if the source file exists
        if [[ -f "$source_file" ]]; then
            # Construct the destination file path
            dest_file="${destination_dir}/${run}-${fold}-predictions.txt"

            # Copy and rename the file
            cp "$source_file" "$dest_file"
            echo "Copied $source_file to $dest_file"
        else
            echo "File not found: $source_file"
        fi
    done
done
