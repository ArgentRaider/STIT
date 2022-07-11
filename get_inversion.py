from utils.models_utils import initialize_e4e_wplus
from torchvision import transforms
from configs import global_config
import PIL
import PIL.Image
import numpy as np
import tqdm

class e4e_inversion_module:
    def __init__(self) -> None:
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.e4e_image_transform = transforms.Resize((256, 256))
        self.e4e_inversion_net = initialize_e4e_wplus()

    def load_image(self, img_path):
        img = PIL.Image.open(img_path).convert('RGB')
        img = self.img_transform(img)
        return img

    def get_inversion(self, image):
        new_image = self.e4e_image_transform(image).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                        input_code=False)
        return w

import os
import click

@click.command()
@click.option('-i', '--input_path', type=str, required=True)
@click.option('-o', '--output_path', type=str, required=True)
@click.option('--is_affect_data', type=bool, default=False)

def _main(input_path, output_path, is_affect_data):
    e4e = e4e_inversion_module()
    
    # data_path = '../../AffectNet/train_set/'
    data_path = input_path

    w_vectors = None
    img_name_list = []

    if not is_affect_data:
        img_name_list = os.listdir(data_path)
        img_name_list.sort()
    else:
        inx = np.load(data_path+'index.npy')
        img_path = 'images/'
        for i in range(30000):
            img_name_list.append(img_path + f'{inx[i]}.jpg')
        # if False and os.path.exists(data_path + 'e4e_w_plus.npy'):
        #     w_vectors = np.load(data_path+'e4e_w_plus.npy')

    for img_name in tqdm.tqdm(img_name_list):
        # img = e4e.load_image(f'rand_data/imgs/{i:0>5}.jpg')

        img = e4e.load_image(os.path.join(data_path,img_name))
        w = e4e.get_inversion(img)
        w = w.detach().cpu().numpy().astype(np.float32)
        if w_vectors is None:
            w_vectors = w
        else:
            w_vectors = np.concatenate([w_vectors, w], axis=0)
    print(w_vectors.shape)
    # np.save('rand_data/w_plus_vectors.npy', w_vectors)
    np.save(os.path.join(output_path, 'e4e_w_plus.npy'), w_vectors)


if __name__ == "__main__":
    _main()