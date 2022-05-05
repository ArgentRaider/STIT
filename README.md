# STIT - Stitch it in Time

For details of the original README, see [link](https://github.com/rotemtzaban/STIT/blob/main/README.md) .

## Data requirements

A video should be splitted into individual frame images and put in a single directory, as mentioned in [link](https://github.com/rotemtzaban/STIT#splitting-videos-into-frames) .
It's advised to put frame images in 'data/{name}/', where '{name}' would be used to store the checkpoint after PTI tuning through scripts. For example: data/someone/00001.jpeg.

## Control method of interactive_amplification.py

- 'z'/'x': pick the previous/next frame
- 'a'/'d': change the start of layers to be edited
- 'w'/'s': change the end of layers to be edited
- 'q'/'e': modify the scale factor for amplification