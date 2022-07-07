# STIT - Stitch it in Time

For details of the original README, see [link](https://github.com/rotemtzaban/STIT/blob/main/README.md) .

## Data requirements

A video should be splitted into individual frame images and put in a single directory, as mentioned in [link](https://github.com/rotemtzaban/STIT#splitting-videos-into-frames) .
It's advised to put frame images in 'data/{name}/', where '{name}' would be used to store the checkpoint after PTI tuning through scripts. For example: data/someone/00001.jpeg.

## Training

Run train.py to get a tuned Generator for your dataset and the inversions from the original StyleGAN generator and tuned generator.

## Crop Data

Before feeding images to the inversion network, we need to align and crop the images first.
'crop_datasets.py' can generate the cropped images after training.
You could also run 'crop.sh' if you have trained on the dataset and saved the checkpoint.

## Control of interactive_(amplification/edit/transfer).py

- 'z'/'x': pick the previous/next frame
- 'a'/'d': change the start of layers to be edited
- 'w'/'s': change the end of layers to be edited
- 'q'/'e': modify the scale factor for amplification
- 'p'    : export the video containing the original frames and edited frames


## Other codes

- 'interactive_amplification_pivots.py': Run amplification on the original StyleGAN generator and any inverted latent codes. I use it to test original e4e space and pSp space.
- (Not Completed) 'get_style_directions.py': Generate directions in Style Space for certain text descriptors, based on the vanilla StyleGAN generator.

