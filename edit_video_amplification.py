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

import models.seg_model_2
from configs import global_config, hyperparameters, paths_config
from edit_video import save_image
from editings.latent_editor import LatentEditor
from utils.alignment import crop_faces_by_quads, calc_alignment_coefficients
from utils.data_utils import make_dataset
from utils.edit_utils import add_texts_to_image_vertical, paste_image, paste_image_mask
from utils.image_utils import concat_images_horizontally, tensor2pil
from utils.models_utils import load_generators, load_old_G, load_from_pkl_model


@click.command()
@click.option('-r', '--run_name', type=str, default=None)
@click.option('-p', '--pivots_path', type=str, default=None)
@click.option('-en', '--edit_name', type=str, default=None, multiple=True)
@click.option('-s', '--scale', type=float, default=1.5)
@click.option('--style_clip', type=bool, default=False)
@click.option('--beta', type=float, default=0.1)
@click.option('--edit_layers_start', type=int, default=0)
@click.option('--edit_layers_end', type=int, default=18)
@click.option('--orig_pivot_type', type=str, default='first')


def _main(run_name, pivots_path, edit_name, scale, style_clip, beta, edit_layers_start, edit_layers_end, orig_pivot_type):
    if not run_name is None:
        gen, orig_gen, pivots, quads = load_generators(run_name)
    elif not pivots_path is None:
        gen = load_from_pkl_model( load_old_G() )
        pivots = np.load(pivots_path)
        pivots = torch.Tensor(pivots).to(global_config.device)

        run_name = os.path.basename(pivots_path)
        dot_index = run_name.rfind('.')
        run_name = run_name[:dot_index]

    else:
        print("Argument '-r' or '-p' must be specified!")
        return


    latent_editor = LatentEditor()

    orig_pivot_type = {'type': orig_pivot_type}

        
    edit_range = (scale,scale,1)
    if not style_clip:
        edits, is_style_input = latent_editor.get_amplification_edits(pivots, edit_name, edit_range, edit_layers_start, edit_layers_end, orig_pivot_type)
    else:
        edits, is_style_input = latent_editor.get_style_clip_amplification_edits(pivots, edit_name, edit_range, gen, beta=beta, origin_pivot_type=orig_pivot_type)

    edits_list, direction, factor = edits[0]

    export_path = 'export_videos/'
    if style_clip:
        edit_name = 'style_' + edit_name[0]
    else:
        edit_name = edit_name[0]
    export_name = f'amplification_{run_name}_{scale:.1f}_{orig_pivot_type["type"]}_{edit_name}_{edit_layers_start}-{edit_layers_end}.mp4'
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    frames = []
    for fj in tqdm(range(len(pivots))):
        orig_frame = tensor2pil( gen.synthesis(pivots[fj][None], noise_mode='const', force_fp32=True) )
        # orig_frame = crops[fj]

        if is_style_input:
            edited_latent = [style[fj][None] for style in edits_list]
        else:
            edited_latent = edits_list[fj][None]
        
        edited_frame = tensor2pil( gen.synthesis.forward(edited_latent, style_input=is_style_input, noise_mode='const', force_fp32=True) )
        orig_frame = orig_frame.resize((512, 512))
        edited_frame = edited_frame.resize((512, 512))
        new_frame = Image.new(orig_frame.mode, (1024, 512))
        new_frame.paste(orig_frame, (0, 0))
        new_frame.paste(edited_frame, (512, 0))
        frames.append(new_frame)
    imageio.mimwrite(os.path.join(export_path, export_name), frames, fps=15, output_params=['-vf', 'fps=15'])
    print("Video exported to", os.path.join(export_path, export_name))



if __name__ == "__main__":
    _main()