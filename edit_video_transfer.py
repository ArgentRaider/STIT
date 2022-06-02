import copy
import io
import json
import os
from collections import defaultdict

import click
import imageio
import torch
import torchvision.transforms.functional
import numpy as np
from PIL import Image, ImageChops
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm, trange
import cv2

import models.seg_model_2
from configs import hyperparameters, paths_config
from edit_video import save_image
from editings.latent_editor import LatentEditor
from utils.alignment import crop_faces_by_quads, calc_alignment_coefficients
from utils.data_utils import make_dataset
from utils.edit_utils import add_texts_to_image_vertical, paste_image, paste_image_mask
from utils.image_utils import concat_images_horizontally, tensor2pil
from utils.models_utils import load_generators
from utils.morphology import dilation

@click.command()
@click.option('-rs', '--run_name_src', type=str, required=True)
@click.option('-rd', '--run_name_dst', type=str, required=True)
@click.option('-o', '--output_path', type=str, required=True)
@click.option('-ot', '--origin_type', type=str, required=True)
@click.option('--edit_layers_start', type=int, default=0)
@click.option('--edit_layers_end', type=int, default=18)
@click.option('--min_exp_weight_path_src', type=str, default=None)
@click.option('--min_exp_weight_path_dst', type=str, default=None)


def _main(run_name_src, run_name_dst, output_path, origin_type, edit_layers_start, edit_layers_end, min_exp_weight_path_src, min_exp_weight_path_dst):
    image_size = 1024
    input_folder_src = f'data_crop/{run_name_src}'
    orig_files_src = make_dataset(input_folder_src)
    input_folder_dst = f'data_crop/{run_name_dst}'
    orig_files_dst = make_dataset(input_folder_dst)
    # segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    # segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    gen_src, _, pivots_src, quads_src = load_generators(run_name_src)
    # crops_src, orig_images_src = crop_faces_by_quads(image_size, orig_files_src, quads_src)
    gen_dst, _, pivots_dst, quads_dst = load_generators(run_name_dst)
    # crops_dst, orig_images_dst = crop_faces_by_quads(image_size, orig_files_dst, quads_dst)

    orig_images_src = []
    crops_src = []
    for (_, path) in orig_files_src:
        orig_image = Image.open(path)
        orig_images_src.append(orig_image)
        crops_src.append(orig_image)
    orig_images_dst = []
    crops_dst = []
    for (_, path) in orig_files_dst:
        orig_image = Image.open(path)
        orig_images_dst.append(orig_image)
        crops_dst.append(orig_image)

    latent_editor = LatentEditor()

    if origin_type == 'mean':
        origin_pivot_type={'type': 'mean'}
    elif origin_type == 'first':
        origin_pivot_type={'type': 'first'}
    elif origin_type == 'min':
        origin_pivot_type={'type': 'min'}
        min_index_src = np.load(min_exp_weight_path_src)['min_index']
        min_index_dst = np.load(min_exp_weight_path_dst)['min_index']
        origin_pivot_type['src_min_index'] = min_index_src
        origin_pivot_type['dst_min_index'] = min_index_dst
    elif origin_type == 'min_weight':
        origin_pivot_type={'type': 'min_weight'}
        weight_src = np.load(min_exp_weight_path_src)['weight']
        origin_pivot_type['src_weight'] = weight_src
        weight_dst = np.load(min_exp_weight_path_dst)['weight']
        origin_pivot_type['dst_weight'] = weight_dst
    
    edits, is_style_input = latent_editor.get_transfer_edits(pivots_src, [pivots_dst], edit_layers_start, edit_layers_end, origin_pivot_type)

    for edits_list, direction, factor in edits:
        for i in \
                tqdm(range(len(orig_images_src))):
            # w_interp = pivots[i][None]

            w_edit_interp = edits_list[i][None]

            edited_tensor = gen_dst.synthesis.forward(w_edit_interp, style_input=is_style_input, noise_mode='const',
                                                  force_fp32=True)
            edited_image = tensor2pil(edited_tensor)


            # folder_name = f'{run_name}_{edit_type}_{factor}_{edit_name}_{edit_layers_start}_{edit_layers_end}'
            # frames_dir = os.path.join(output_folder, folder_name)
            frames_dir = output_path
            os.makedirs(frames_dir, exist_ok=True)
            save_image(edited_image, os.path.join(frames_dir, f'edit_{i:04d}.jpeg'))


if __name__ == "__main__":
    _main()