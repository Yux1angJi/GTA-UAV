import os
from PIL import Image


def is_tile_transparent(tile):
    """检查瓦片是否完全透明"""
    extrema = tile.getextrema()
    if extrema[3][0] == 0 and extrema[3][1] == 0:
        return True
    return False

def create_tiles(image, tile_size, output_dir, level):
    num_tiles = 2 ** level
    tile_pixel_size = tile_size * num_tiles
    img = image.resize((tile_pixel_size, tile_pixel_size), Image.Resampling.LANCZOS)
    
    level_dir = os.path.join(output_dir, f'level_{level}')
    os.makedirs(level_dir, exist_ok=True)

    for x in range(num_tiles):
        for y in range(num_tiles):
            left = x * tile_size
            upper = y * tile_size
            right = left + tile_size
            lower = upper + tile_size

            tile = img.crop((left, upper, right, lower))

            if is_tile_transparent(tile):
                print(f'Skipped transparent tile at {level}/{x}_{y}.png')
                continue

            tile_path = os.path.join(level_dir, f'{level}_{x}_{y}.png')
            tile.save(tile_path)
            # print(f'Saved {tile_path}')

def slice_image(input_path, output_dir, tile_size=256, max_level=7):
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(input_path)
    width, height = img.size

    assert width == height == 24576, "Input image must be 24576x24576 pixels"

    for level in range(max_level + 1):
        create_tiles(img, tile_size, output_dir, level)

if __name__ == "__main__":
    input_image_path = '../satellite/GTAV_satellite_square.png'
    output_directory = '../satellite/tiles'
    slice_image(input_image_path, output_directory)
