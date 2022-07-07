if [ -n "$3" ]; then
    echo "RUN_NAME: $1"
    echo "FACTOR: $2"
    echo "PIVOT NAME: $3"

    if [ -n "$5" ]; then
        echo "OUTPUT_PATH: data_crop/$1_amplification_$2_$3_$4_$5"
        echo "Start Layer: $4"
        echo "End Layer: $5"
        python edit_video_amplification.py --input_folder data_crop/$1/ --input_crop True --output_folder data_crop/$1_amplification_$2_$3_$4_$5 --run_name $1 --edit_type amplification --pivot_name $3 --edit_range $2 $2 1 --output_video False --output_frames --edit_layers_start $4 --edit_layers_end $5 --min_exp_weight_path data_crop_codes/$1/min_exp_weight.npz
    else
        echo "OUTPUT_PATH: data_crop/$1_amplification_$2_$3_None_None"
        python edit_video_amplification.py --input_folder data_crop/$1/ --input_crop True --output_folder data_crop/$1_amplification_$2_$3_None_None --run_name $1 --edit_type amplification --pivot_name $3 --edit_range $2 $2 1 --output_video False --output_frames --min_exp_weight_path data_crop_codes/$1/min_exp_weight.npz
    fi
    
    
else
    echo "bash amplificate.sh <RUN_NAME> <FACTOR> <EDIT_NAME> [Start Layer] [End Layer]"
fi