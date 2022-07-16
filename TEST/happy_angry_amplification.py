import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
import torch
import tqdm
from PIL import Image
import imageio

from configs import global_config, hyperparameters, paths_config
from utils.models_utils import load_old_G, load_from_pkl_model
from editings import styleclip_global_utils
from utils.image_utils import tensor2pil
from utils.edit_utils import get_affine_layers, to_styles, w_to_styles


def _main():
    factor = 10.0

    w = np.load('w_vectors/e4e_yukun_happy_anger.npy')
    w = torch.tensor(w).to(global_config.device)
    gen = load_from_pkl_model(load_old_G())
    affine_layers = get_affine_layers(gen.synthesis)
    styles = w_to_styles(w, affine_layers)

    happy_direction = styleclip_global_utils.get_direction('face', 'happy', 0.1)
    happy_style_direction = to_styles(happy_direction, affine_layers)
    angry_direction = styleclip_global_utils.get_direction('face', 'angry', 0.1)
    angry_style_direction = to_styles(angry_direction, affine_layers)
    for i in range(len(happy_style_direction)):
        len_i_happy = torch.norm(happy_style_direction[i], 2)
        if len_i_happy > 1e-4:
            happy_style_direction[i] /= len_i_happy
        happy_style_direction[i] = happy_style_direction[i].float()
        len_i_angry = torch.norm(angry_style_direction[i], 2)
        if len_i_angry > 1e-4:
            angry_style_direction[i] /= len_i_angry
        angry_style_direction[i] = angry_style_direction[i].float()
    
    edited_styles = []
    for i in range(len(styles)):
        dist = styles[i] - styles[i][0]
        edited_style = styles[i].clone()
        for j in range(dist.shape[0]):
            happy_component = torch.dot(dist[j], happy_style_direction[i]).item()
            angry_component = torch.dot(dist[j], angry_style_direction[i]).item()
            if happy_component > 10 * angry_component:
                amplify_component = happy_component
                amplify_direction = happy_style_direction[i] 
                factor = 10.0
            else:
                amplify_component = angry_component
                amplify_direction = angry_style_direction[i]
                factor = 10.0
            
            amplify_direction_len = torch.norm(amplify_direction, 2).item()
            if amplify_direction_len < 1e-4:
                dist[j] = torch.zeros(dist[j].shape, dtype=dist.dtype, device=dist.device)
            else:
                dist[j] = amplify_component * amplify_direction
                ...
            
            edited_style[j] += (factor - 1) * dist[j]
        edited_styles.append(edited_style)
        
    frames = []
    for fj in tqdm.tqdm(range(w.shape[0])):
        orig_frame = tensor2pil( gen.synthesis(w[fj][None], noise_mode='const', force_fp32=True) )

        edited_latent = [style[fj][None] for style in edited_styles]
        edited_frame = tensor2pil( gen.synthesis.forward(edited_latent, style_input=True, noise_mode='const', force_fp32=True) )

        orig_frame = orig_frame.resize((512, 512))
        edited_frame = edited_frame.resize((512, 512))
        new_frame = Image.new(orig_frame.mode, (1024, 512))
        new_frame.paste(orig_frame, (0, 0))
        new_frame.paste(edited_frame, (512, 0))
        frames.append(new_frame)
    export_path = 'TEST/happy_angry.mp4'
    imageio.mimwrite(export_path, frames, fps=15, output_params=['-vf', 'fps=15'])
    print("Video exported to", export_path)
        

if __name__ == "__main__":
    _main()