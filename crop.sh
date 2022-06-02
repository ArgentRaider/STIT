if [ -n "$1" ]; then
    echo "RUN_NAME: $1"
    python crop_datasets.py --input_folder data/$1/ --output_folder data_crop/$1 --run_name $1
else
    echo "bash crop.sh <RUN_NAME>"
fi