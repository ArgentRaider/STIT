from pyrsistent import get_in
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
if __name__ == "__main__":
    e4e = e4e_inversion_module()
    
    data_path = '../../AffectNet/train_set/'
    inx = np.load(data_path+'index.npy')
    img_path = data_path + 'images/'
    if False and os.path.exists(data_path + 'e4e_w_plus.npy'):
        w_vectors = np.load(data_path+'e4e_w_plus.npy')
    else:
        w_vectors = None

    for ti in tqdm.trange(0,30000):
        # img = e4e.load_image(f'rand_data/imgs/{i:0>5}.jpg')
        i = inx[ti]
        img = e4e.load_image(img_path + f'{i}.jpg')
        w = e4e.get_inversion(img)
        w = w.detach().cpu().numpy().astype(np.float32)
        if w_vectors is None:
            w_vectors = w
        else:
            w_vectors = np.concatenate([w_vectors, w], axis=0)
    print(w_vectors.shape)
    # np.save('rand_data/w_plus_vectors.npy', w_vectors)
    np.save(data_path + 'e4e_w_plus.npy', w_vectors)
