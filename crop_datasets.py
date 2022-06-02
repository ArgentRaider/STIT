from email.policy import default
import click
import os
from utils.data_utils import make_dataset
from utils.models_utils import load_generators
from utils.alignment import crop_faces_by_quads, crop_faces
from tqdm import trange

@click.command()
@click.option('-i', '--input_folder', type=str, help='Path to (unaligned) images folder', required=True)
@click.option('-o', '--output_folder', type=str, help='Path to output folder', required=True)
@click.option('-r', '--run_name', type=str, default = None)

def main(input_folder, output_folder, run_name):
    os.makedirs(output_folder, exist_ok=True)

    orig_files = make_dataset(input_folder)

    image_size = 1024
    if not run_name is None:
        print("Loading saved quads...")
        gen, orig_gen, pivots, quads = load_generators(run_name)
        crops, orig_images = crop_faces_by_quads(image_size, orig_files, quads)
    else:
        print("Detecting faces...")
        crops, orig_images, quads = crop_faces(image_size, orig_files, scale=1.0,
                                           center_sigma=1.0, xy_sigma=3.0, use_fa=True)

    for i in trange(len(crops)):
        output_path = os.path.join(output_folder, '{:0>4}.jpeg'.format(i))
        crops[i].save(output_path)

if __name__ == "__main__":
    main()