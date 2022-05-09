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

from UI import UI

@click.command()
@click.option('-rs', '--run_name_src', type=str, required=True)
@click.option('-rd', '--run_name_dst', type=str, required=True)
@click.option('--edit_layers_start', type=int, default=0)
@click.option('--edit_layers_end', type=int, default=18)


def _main(run_name_src, run_name_dst, edit_layers_start, edit_layers_end):
    image_size = 1024
    input_folder_src = f'data/{run_name_src}'
    orig_files_src = make_dataset(input_folder_src)
    input_folder_dst = f'data/{run_name_dst}'
    orig_files_dst = make_dataset(input_folder_dst)
    segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    gen_src, _, pivots_src, quads_src = load_generators(run_name_src)
    crops_src, orig_images_src = crop_faces_by_quads(image_size, orig_files_src, quads_src)
    gen_dst, _, pivots_dst, quads_dst = load_generators(run_name_dst)
    crops_dst, orig_images_dst = crop_faces_by_quads(image_size, orig_files_dst, quads_dst)

    latent_editor = LatentEditor()
    ui = UI(windowName=f'Transfer-{run_name_src}-{run_name_dst}')

    fi = 0
    is_playing = False
    use_mean = False
    window_should_close = False
    while not window_should_close:
        # print(fi)
        
        edits, is_style_input = latent_editor.get_transfer_edits(pivots_src, [pivots_dst], edit_layers_start, edit_layers_end, use_mean)

        edits_list, direction, factor = edits[0]

        src_tensor = gen_src.synthesis(pivots_src[fi][None], noise_mode='const', force_fp32=True)
        src_image = tensor2pil(src_tensor)
        # orig_image = crops[fi]
        
        w_transfer = edits_list[fi][None]
        transferred_tensor = gen_dst.synthesis.forward(w_transfer, style_input=False, noise_mode='const',
                                                force_fp32=True)
        transferred_image = tensor2pil(transferred_tensor)        

        src_image = np.array(src_image)[:, :, ::-1]
        src_image = cv2.resize(src_image, (512, 512))
        transferred_image = np.array(transferred_image)[:, :, ::-1]
        transferred_image = cv2.resize(transferred_image, (512, 512))

        textList = []
        textList.append(f'Frame: {fi}')
        textList.append(f'Edit Layers: {edit_layers_start}-{edit_layers_end}')
        if not use_mean:
            textList.append('First or Mean: First')
        else:
            textList.append('First or Mean: Mean')

        ui.display(textList, src_image, transferred_image)

        if is_playing:
            ret = cv2.waitKey(1000/25)
        else:
            ret = cv2.waitKey()
        
        if ret == 27: #'ESC'
            window_should_close = True
        elif ret == 32: # space
            # is_playing = not is_playing
            pass
        elif ret == 122: # 'z'
            is_playing = False
            fi -= 1
            if fi < 0:
                fi = 0
        elif ret == 120: # 'x'
            is_playing = False
            fi += 1
            if fi >= len(pivots_src):
                fi = len(pivots_src) - 1
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
        elif ret == 112: # 'p' export video
            # maybe I should also write the config to a json file or something
            export_path = 'export_videos/'
            export_name = f'transfer_{run_name_src}_{run_name_dst}.mp4'
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            
            frames = []
            for fj in tqdm(range(len(pivots_src))):
                orig_frame = tensor2pil( gen_src.synthesis(pivots_src[fj][None], noise_mode='const', force_fp32=True) )
                edited_frame = tensor2pil( gen_dst.synthesis.forward(edits_list[fj][None], style_input=False, noise_mode='const', force_fp32=True) )
                orig_frame = orig_frame.resize((512, 512))
                edited_frame = edited_frame.resize((512, 512))
                new_frame = Image.new(orig_frame.mode, (1024, 512))
                new_frame.paste(orig_frame, (0, 0))
                new_frame.paste(edited_frame, (512, 0))
                frames.append(new_frame)
            imageio.mimwrite(os.path.join(export_path, export_name), frames, fps=25, output_params=['-vf', 'fps=25'])
            print("Video exported to", os.path.join(export_path, export_name))

        if is_playing: # inactive feature for now
            fi += 1
            if fi >= len(pivots_src):
                fi = len(pivots_src) - 1


if __name__ == "__main__":
    _main()