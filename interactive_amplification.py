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
@click.option('-r', '--run_name', type=str, required=True)
@click.option('--edit_layers_start', type=int, default=0)
@click.option('--edit_layers_end', type=int, default=18)


def _main(run_name, edit_layers_start, edit_layers_end):
    input_folder = f'data/{run_name}'
    orig_files = make_dataset(input_folder)
    segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    gen, orig_gen, pivots, quads = load_generators(run_name)
    image_size = 1024

    crops, orig_images = crop_faces_by_quads(image_size, orig_files, quads)

    latent_editor = LatentEditor()

    edit_name='amplification'

    fi = 0
    scale = 2
    is_playing = False
    use_mean = False
    display = np.zeros((512, 3*512, 3), dtype=np.uint8)
    while True:
        print(fi)
        
        edit_range = (scale,scale,1)
        pivots_copy = pivots.clone()
        edits, is_style_input = latent_editor.get_amplification_edits(pivots_copy, edit_range, edit_layers_start, edit_layers_end, use_mean)

        edits_list, direction, factor = edits[0]

        w_interp = pivots[fi][None]
        inversion = gen.synthesis(w_interp, noise_mode='const', force_fp32=True)
        orig_image = tensor2pil(inversion)
        
        w_edit_interp = edits_list[fi][None]
        edited_tensor = gen.synthesis.forward(w_edit_interp, style_input=False, noise_mode='const',
                                                force_fp32=True)
        edited_image = tensor2pil(edited_tensor)

        edited_image = np.array(edited_image)[:, :, ::-1]
        edited_image = cv2.resize(edited_image, (512, 512))

        orig_image = np.array(orig_image)[:, :, ::-1]
        display[:, 512:2*512, :] = cv2.resize(orig_image, (512, 512))
        display[:, 2*512:3*512, :] = edited_image
        display[:, :512, :] = 0

        cv2.putText(display, f'Frame: {fi}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.putText(display, f'Scale: {scale}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.putText(display, f'Edit Layers: {edit_layers_start}-{edit_layers_end}', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        if not use_mean:
            cv2.putText(display, f'First or Mean: First', (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        else:
            cv2.putText(display, f'First or Mean: Mean', (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)


        cv2.imshow('test', display)
        if is_playing:
            ret = cv2.waitKey(1000/25)
        else:
            ret = cv2.waitKey()
            print(ret)
        
        if ret == 27: #'ESC'
            break
        elif ret == 32: # space
            pass
        elif ret == 122: # 'z'
            is_playing = False
            fi -= 1
            if fi < 0:
                fi = 0
        elif ret == 120: # 'x'
            is_playing = False
            fi += 1
            if fi >= len(orig_images):
                fi = len(orig_images) - 1
        elif ret == 113: # 'q'
            scale -= 0.1
            pass
        elif ret == 101: # 'e'
            scale += 0.1
            pass
        elif ret == 97:  # 'a'
            edit_layers_start -= 1
            if edit_layers_start < 0:
                edit_layers_start = 0
        elif ret == 100: # 'd'
            edit_layers_start += 1
            if edit_layers_start >= edit_layers_end:
                edit_layers_start = edit_layers_end - 1
        elif ret == 115: # 's'
            edit_layers_end -= 1
            if edit_layers_end <= edit_layers_start:
                edit_layers_end = edit_layers_start + 1
        elif ret == 119: # 'w'
            edit_layers_end += 1
            if edit_layers_end > 18:
                edit_layers_end = 18
        elif ret == 99:  # 'c'
            use_mean = not use_mean

        if is_playing: # inactive feature for now
            fi += 1
            if fi >= len(orig_images):
                fi = len(orig_images) - 1


if __name__ == "__main__":
    _main()