import os
from collections import defaultdict

import click
import numpy as np
from PIL import Image, ImageChops
from torch.nn import functional as F

from configs import global_config, hyperparameters, paths_config
from utils.models_utils import load_generators, load_old_G, load_from_pkl_model

from editings import styleclip_global_utils
from utils.edit_utils import get_affine_layers, load_stylespace_std, to_styles, w_to_styles

import pickle

def get_style_directions(direction_name):
    beta = 0.1
    gen = load_from_pkl_model( load_old_G() )

    affine_layers = get_affine_layers(gen.synthesis)
    edit_directions = styleclip_global_utils.get_direction('face', direction_name, beta)
    # print(edit_directions.shape)
    edit = to_styles(edit_directions, affine_layers)

    print(edit[14])

    # for i in range(len(edit)):
    #     edit[i] = edit[i].cpu()

    return edit

@click.command()
@click.option('-dn', '--direction_name', type=str)
def _main(direction_name):
    style_direction = get_style_directions(direction_name)
    save_path = 'editings/style_directions/'
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{direction_name}.pkl')
    f = open(save_path, 'wb')
    print("Dumping to " + save_path)
    pickle.dump(style_direction, f)
    f.close()

    # Test part
    if True:
        print(style_direction[14])
        f = open(save_path, 'rb')
        style_direction_loaded = pickle.load(f)
        f.close()
        print(style_direction_loaded[14])

if __name__ == "__main__":
    _main()