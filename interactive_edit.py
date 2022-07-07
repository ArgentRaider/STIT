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

EXPORT_FORMAT = "IMAGE"

@click.command()
@click.option('-r', '--run_name', type=str, required=True)
@click.option('-en', '--edit_name', type=str, required=True)
@click.option('--edit_layers_start', type=int, default=0)
@click.option('--edit_layers_end', type=int, default=18)

def _main(run_name, edit_name, edit_layers_start, edit_layers_end):
    image_size = 1024
    # input_folder = f'data/{run_name}'
    # orig_files = make_dataset(input_folder)
    # segmentation_model = models.seg_model_2.BiSeNet(19).eval().cuda().requires_grad_(False)
    # segmentation_model.load_state_dict(torch.load(paths_config.segmentation_model_path))

    gen, orig_gen, pivots, quads = load_generators(run_name)
    ## The unedited images could be pre-generated, yet they consume too much memory...
    # gen_images = []
    # for fi in range(len(pivots)):
    #     gen_image = gen.synthesis(pivots[fi][None], noise_mode='const', force_fp32=True)
    #     gen_images.append(tensor2pil(gen_image))

    # crops, orig_images = crop_faces_by_quads(image_size, orig_files, quads)

    latent_editor = LatentEditor()
    ui = UI(windowName=f'{edit_name}-{run_name}')
    edit_name = [edit_name]

    fi = 0
    scale = 1
    is_playing = False
    window_should_close = False
    status_changed = True
    while not window_should_close:
        # print(fi)
        
        if status_changed:
            edit_range = (scale,scale,1)
            edits, is_style_input = latent_editor.get_interfacegan_edits(pivots, edit_name, edit_range, edit_layers_start, edit_layers_end)
            status_changed = False

        edits_list, direction, factor = edits[0]

        orig_tensor = gen.synthesis(pivots[fi][None], noise_mode='const', force_fp32=True)
        orig_image = tensor2pil(orig_tensor)
        # orig_image = crops[fi]
        
        w_edit_interp = edits_list[fi][None]
        edited_tensor = gen.synthesis.forward(w_edit_interp, style_input=False, noise_mode='const',
                                                force_fp32=True)
        edited_image = tensor2pil(edited_tensor)

        edited_image = np.array(edited_image)[:, :, ::-1]
        edited_image = cv2.resize(edited_image, (512, 512))

        orig_image = np.array(orig_image)[:, :, ::-1]
        orig_image = cv2.resize(orig_image, (512, 512))

        textList = []
        textList.append(f'Frame: {fi}')
        textList.append(f'Scale: {scale}')
        textList.append(f'Edit Layers: {edit_layers_start}-{edit_layers_end}')


        ui.display(textList, orig_image, edited_image)

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
            if fi >= len(pivots):
                fi = len(pivots) - 1
        elif ret == 113: # 'q'
            scale -= 0.1
            status_changed = True
        elif ret == 101: # 'e'
            scale += 0.1
            status_changed = True
        elif ret == 97:  # 'a'
            if edit_layers_start > 0:
                edit_layers_start -= 1
                status_changed = True
        elif ret == 100: # 'd'
            if edit_layers_start < edit_layers_end - 1:
                edit_layers_start += 1
                status_changed = True
        elif ret == 115: # 's'
            if edit_layers_end > edit_layers_start + 1:
                edit_layers_end -= 1
                status_changed = True
        elif ret == 119: # 'w'
            if edit_layers_end < 18:
                edit_layers_end += 1
                status_changed = True
        
        elif ret == 112: # 'p' export video
            if EXPORT_FORMAT == "VIDEO":
                export_path = 'export_videos/'
                export_name = f'{edit_name[0]}_{run_name}_{scale}.mp4'
                os.makedirs(export_path, exist_ok=True)
                
                frames = []
                for fj in tqdm(range(len(pivots))):
                    orig_frame = tensor2pil( gen.synthesis(pivots[fj][None], noise_mode='const', force_fp32=True) )
                    edited_frame = tensor2pil( gen.synthesis.forward(edits_list[fj][None], style_input=False, noise_mode='const', force_fp32=True) )
                    orig_frame = orig_frame.resize((512, 512))
                    edited_frame = edited_frame.resize((512, 512))
                    new_frame = Image.new(orig_frame.mode, (1024, 512))
                    new_frame.paste(orig_frame, (0, 0))
                    new_frame.paste(edited_frame, (512, 0))
                    frames.append(new_frame)
                imageio.mimwrite(os.path.join(export_path, export_name), frames, fps=25, output_params=['-vf', 'fps=25'])
                print("Video exported to", os.path.join(export_path, export_name))
            elif EXPORT_FORMAT == "IMAGE":
                export_path = 'export_images/'
                export_name = f'{edit_name[0]}_{run_name}_{scale}/'
                os.makedirs(export_path+export_name, exist_ok=True)

                for fj in tqdm(range(len(pivots))):
                    edited_frame = tensor2pil( gen.synthesis.forward(edits_list[fj][None], style_input=False, noise_mode='const', force_fp32=True) )
                    imageio.imwrite(export_path+export_name+f'{fj:0>4}.jpg', edited_frame)
                print("Images exported to", export_path+export_name)

        if is_playing: # inactive feature for now
            fi += 1
            if fi >= len(pivots):
                fi = len(pivots) - 1


if __name__ == "__main__":
    _main()