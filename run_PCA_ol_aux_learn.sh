#!/bin/bash

# Check if a file name was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <main_task_file.csv>"
    exit 1
fi

main_task_file="$1"
base_name=$(basename "$main_task_file" .csv) # Extract base name without extension

# Iterate from 0 to 29, giving 30 different seeds
for i in {0..29}
do
    echo "Running iteration with seed $i"
    python PCA_MLC_ol_aux.py --mccv=$i --main_task_file="$main_task_file"
done

echo "All iterations completed."

# Concatenate all files into one, including the header from the first file
echo "AUROC,AUPR" > "./output/PCA_ol_aux/ol_aux_combined_${base_name}.csv"
for i in {0..29}
do
    # Skip header (first line) for all files except the first one
    if [ $i -eq 0 ]; then
        cat "./output/MCCV_PCA_ol_aux_${base_name}_$i.csv" >> "./output/PCA_ol_aux/ol_aux_combined_${base_name}.csv"
    else
        tail -n +2 "./output/MCCV_PCA_ol_aux_${base_name}_$i.csv" >> "./output/PCA_ol_aux/ol_aux_combined_${base_name}.csv"
    fi
done

# Remove the individual files
for i in {0..29}
do
    rm "./output/MCCV_PCA_ol_aux_${base_name}_$i.csv"
done

echo "Files concatenated and original files removed."


# #!/bin/bash

# # Iterate from 0 to 29, giving 30 different seeds
# for i in {0..29}
# do
#     echo "Running iteration with seed $i"
#     python PCA_MLC_ol_aux_prototype.py --mccv=$i --main_task_file='GSE164458_paclitaxel.csv'
# done

# echo "All iterations completed."
