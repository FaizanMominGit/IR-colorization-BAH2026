import os
import argparse
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from utils.file_utils import validate_extension, read_tif_image, write_tif_image
from utils.visualization import percentile_stretch

def merge_rgb_bands(red_path, green_path, blue_path, output_rgb_path):
    validate_extension(red_path)
    validate_extension(green_path)
    validate_extension(blue_path)

    os.makedirs(os.path.dirname(output_rgb_path), exist_ok=True)

    red = read_tif_image(red_path)
    green = read_tif_image(green_path)
    blue = read_tif_image(blue_path)

    if red.ndim == 3: red = red[0]
    if green.ndim == 3: green = green[0]
    if blue.ndim == 3: blue = blue[0]

    rgb_image = np.stack([red, green, blue], axis=0)
    write_tif_image(output_rgb_path, rgb_image, photometric='rgb')
    logger.info(f'Merged RGB bands into {output_rgb_path}')

    rgb_viz = rgb_image.transpose(1, 2, 0)
    rgb_pil_image = percentile_stretch(rgb_viz)
    img = Image.fromarray(rgb_pil_image)

    png_output_dir = os.path.join(os.path.dirname(output_rgb_path), 'png')
    os.makedirs(png_output_dir, exist_ok=True)
    png_output_path = os.path.join(png_output_dir, os.path.splitext(os.path.basename(output_rgb_path))[0] + '.png')
    img.save(png_output_path)
    logger.info(f'Saved visualization PNG to {png_output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge individual Red, Green, Blue bands into a single RGB TIFF.')
    parser.add_argument('red_path', type=str, help='Path to the Red band TIFF file.')
    parser.add_argument('green_path', type=str, help='Path to the Green band TIFF file.')
    parser.add_argument('blue_path', type=str, help='Path to the Blue band TIFF file.')
    parser.add_argument('output_rgb_path', type=str, help='Path to save the merged RGB TIFF file.')

    args = parser.parse_args()

    try:
        merge_rgb_bands(args.red_path, args.green_path, args.blue_path, args.output_rgb_path)
    except Exception as e:
        logger.error(f"Error merging RGB bands: {e}")
        exit(1)
