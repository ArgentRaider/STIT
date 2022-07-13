import sys

sys.path.append('./')
sys.path.append('../')

from utils.image_utils import concat_images_horizontally, tensor2pil

import os
import numpy as np
import torch

from TEST.visualize_direction import NEUTRAL_PIVOTS_PATH, init_neutral_pivots
from utils.models_utils import load_generators, load_old_G, load_from_pkl_model

from editings import styleclip_global_utils
from utils.edit_utils import get_affine_layers, load_stylespace_std, to_styles, w_to_styles

def test_style_directions(orig_w, direction_name, generator):
    neutral_class = 'face'
    target_class = direction_name
    beta = 0.1
    affine_layers = get_affine_layers(generator.synthesis)
    edit_directions = styleclip_global_utils.get_direction(neutral_class, target_class, beta)
    edit = to_styles(edit_directions, affine_layers)

    styles = w_to_styles(orig_w, affine_layers)

    factor = 1
    # print(edit[14])
    pos_edited_styles = [style + factor * edit_direction for style, edit_direction in zip(styles, edit)]
    neg_edited_styles = [style - factor * edit_direction for style, edit_direction in zip(styles, edit)]

    orig_tensor = generator.synthesis.forward(styles, style_input=True, noise_mode='const',
                                                  force_fp32=True)
    pos_edited_tensor = generator.synthesis.forward(pos_edited_styles, style_input=True, noise_mode='const',
                                                  force_fp32=True)
    neg_edited_tensor = generator.synthesis.forward(neg_edited_styles, style_input=True, noise_mode='const',
                                                  force_fp32=True)
    orig_img = tensor2pil(orig_tensor)
    pos_edited_img = tensor2pil(pos_edited_tensor)
    neg_edited_img = tensor2pil(neg_edited_tensor)
    display_img = concat_images_horizontally(orig_img, pos_edited_img, neg_edited_img)
    display_img = display_img.resize((display_img.width // 2, display_img.height // 2))
    display_img.save(os.path.join('TEST/style_direction_visualization', f'{direction_name}.jpg'))

def _main():
    generator = load_from_pkl_model( load_old_G() )
    if not os.path.exists(NEUTRAL_PIVOTS_PATH):
        init_neutral_pivots()
    neutral_pivots_np = np.load(NEUTRAL_PIVOTS_PATH)
    neutral_pivots_cuda = torch.from_numpy(neutral_pivots_np).cuda()
    orig_w = neutral_pivots_cuda[0][None]

    test_style_directions(orig_w, 'happy', generator)

if __name__ == "__main__":
    _main()
