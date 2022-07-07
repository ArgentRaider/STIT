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
from utils.image_utils import concat_images_horizontally, tensor2pil
from utils.models_utils import load_generators, load_old_G, load_from_pkl_model

from editings import styleclip_global_utils
from utils.edit_utils import get_affine_layers, load_stylespace_std, to_styles, w_to_styles

@click.command()
@click.option('-dn', '--direction_name', type=str)
def _main(direction_name):
    beta = 0.2
    gen = load_from_pkl_model( load_old_G() )

    affine_layers = get_affine_layers(gen.synthesis)
    edit_directions = styleclip_global_utils.get_direction('face', direction_name, beta)
    # print(edit_directions.shape)
    edit = to_styles(edit_directions, affine_layers)
    print(len(edit))
    print(edit[0].shape)
    # edit = torch.cat(edit)

    # edit = edit.detach().cpu().numpy()
    # print(edit.shape)
    # np.save(f'editings/w_directions/style_{direction_name}_4.npy', edit)

if __name__ == "__main__":
    _main()