import numpy as np
import torch
import pickle
from PIL import Image
from utils.image_utils import tensor2pil
import os
import tqdm
import imageio
from configs import paths_config

def load_stylegan2(pickle_path='pretrained_models/ffhq.pkl'):
    with open(pickle_path, 'rb') as f:
        G_ema = pickle.load(f)['G_ema'].eval().cuda()
        synthesis = G_ema.synthesis
        mapping = G_ema.mapping
    return mapping, synthesis

def rand_sample(num, dim=512):
    return torch.Tensor(np.random.randn(num, dim)).cuda()

def synthesize_images(output_path, z, mapping_network, synthesis_network, max_batch_size=5):
    total_len = len(z)
    batch_num = (total_len-1) // max_batch_size + 1
    w_vectors = None

    with torch.no_grad():
        for bi in tqdm.trange(batch_num):
            start = bi * max_batch_size
            batch_len = min(max_batch_size, total_len - start)
            batch_z = z[start:start+batch_len]

            w = mapping_network.forward(batch_z, c=18)
            img = synthesis_network.forward(w, noise_mode='const', force_fp32=True)

            w_np = w.detach().cpu().numpy()
            if w_vectors is None:
                w_vectors = w_np
            else:
                w_vectors = np.vstack([w_vectors, w_np])

            for i in range(img.shape[0]):
                img_pil = tensor2pil(img[i, None])
                inx = i + start
                img_pil.save(os.path.join(output_path, f'{inx:0>5}.jpg'))
            
            torch.cuda.empty_cache()
    np.save(os.path.join(output_path, '../w_vectors.npy'), w_vectors)

def gen_rand_data(img_num, stylegan_path, output_path):
    print("Loading StyleGAN2...")
    mapping, synthesis = load_stylegan2(pickle_path=stylegan_path)

    print("Random sampling z...")
    rand_z = rand_sample(img_num)

    print("Synthesizing images...")
    synthesize_images(output_path, rand_z, mapping, synthesis)

if __name__ == "__main__":
    img_num = 10000
    stylegan_path = paths_config.stylegan2_ada_ffhq
    output_path = 'rand_data/imgs/'
    os.makedirs(output_path, exist_ok=True)
    gen_rand_data(img_num, stylegan_path, output_path)
    