if [ -n "$3" ]; then
    echo "SRC_NAME: $1"
    echo "DST_NAME: $2"
    echo "EDIT NAME: $3"

    if [ -n "$5" ]; then
        echo "OUTPUT_PATH: data_crop/$1_$2_transfer_$3_$4_$5"
        echo "Start Layer: $4"
        echo "End Layer: $5"
        python edit_video_transfer.py --run_name_src $1 --run_name_dst $2 --output_path data_crop/$1_$2_transfer_$3_$4_$5 --origin_type $3 --edit_layers_start $4 --edit_layers_end $5 --min_exp_weight_path_src data_crop_codes/$1/min_exp_weight.npz --min_exp_weight_path_dst data_crop_codes/$2/min_exp_weight.npz
    else
        echo "OUTPUT_PATH: data_crop/$1_$2_transfer_$3_None_None"
        python edit_video_transfer.py --run_name_src $1 --run_name_dst $2 --output_path data_crop/$1_$2_transfer_$3_None_None --origin_type $3 --min_exp_weight_path_src data_crop_codes/$1/min_exp_weight.npz --min_exp_weight_path_dst data_crop_codes/$2/min_exp_weight.npz
    fi
    
    
else
    echo "bash amplificate.sh <RUN_NAME> <FACTOR> <EDIT_NAME> [Start Layer] [End Layer]"
fi