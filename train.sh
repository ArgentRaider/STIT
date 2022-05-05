if [ -n "$1" ]; then
    echo "RUN_NAME: $1"
    python train.py --input_folder data/$1/ --output_folder training_results/$1 --run_name $1 --num_pti_steps 80
else
    echo "bash train.sh <RUN_NAME>"
fi