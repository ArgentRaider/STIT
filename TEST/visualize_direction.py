import sys

sys.path.append('./')
sys.path.append('../')
import os

import numpy as np
import torch
import cv2
from utils.models_utils import load_generators, load_old_G, load_from_pkl_model
from utils.image_utils import concat_images_horizontally, tensor2pil

import click

NEUTRAL_PIVOTS_PATH = 'TEST/neutral_pivots.npy'

from get_inversion import e4e_inversion_module
def init_neutral_pivots(save_path = NEUTRAL_PIVOTS_PATH):
    e4e = e4e_inversion_module()
    img_path = 'data_crop/Neutral/0006.jpeg'
    img = e4e.load_image(img_path)
    w_cuda = e4e.get_inversion(img)
    w_np = w_cuda.detach().cpu().numpy().astype(np.float32)
    np.save(save_path, w_np)


@click.command()
@click.option('-dn', '--direction_name', type=str)

def _main(direction_name):
    direction_np = np.load(f'editings/w_directions/{direction_name}.npy')
    direction_cuda = torch.from_numpy(direction_np).cuda()
    for li in range(direction_cuda.shape[0]):
        direction_cuda[li] /= torch.norm(direction_cuda[li], 2).item()

    if not os.path.exists(NEUTRAL_PIVOTS_PATH):
        print("Neutral pivots data not found. Initializing ...")
        init_neutral_pivots()
    
    neutral_pivots_np = np.load(NEUTRAL_PIVOTS_PATH)
    neutral_pivots_cuda = torch.from_numpy(neutral_pivots_np).cuda()

    generator = load_from_pkl_model( load_old_G() )

    orig_w = neutral_pivots_cuda[0][None]
    pos_edited_w = (neutral_pivots_cuda[0] + 3 * direction_cuda)[None]
    neg_edited_w = (neutral_pivots_cuda[0] - 3 * direction_cuda)[None]
    input_tensor = torch.concat([orig_w, pos_edited_w, neg_edited_w])
    images = generator.synthesis(input_tensor, style_input=False, noise_mode='const',
                                                force_fp32=True)
    orig_image = tensor2pil(images[0][None])
    pos_edited_image = tensor2pil(images[1][None]) 
    neg_edited_image = tensor2pil(images[2][None])

    display_img = concat_images_horizontally(orig_image, pos_edited_image, neg_edited_image)
    display_img = display_img.resize( (display_img.width // 2, display_img.height // 2) )
    display_img_np = np.array(display_img)[:, :, ::-1] #RGB to BGR

    cv2.imshow(direction_name, display_img_np)
    cv2.waitKey()

    save_dir = 'TEST/direction_visualization/'
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(save_dir + direction_name + '.jpg', display_img_np)


if __name__ == "__main__":
    _main()