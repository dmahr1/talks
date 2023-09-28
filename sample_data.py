import numpy as np
from PIL import Image

COLORS = [
    (255, 255, 0),  # yellow
    (0, 255, 0),  # green
    (255, 0, 0),  # red
    (0, 0, 0),  # black
    (0, 0, 255),  # blue
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
]


def create_array_of_colored_stripes(
    colors: list[tuple[int, int, int]], col_width: int, gutter_width: int, num_repeats: int = 4
) -> np.ndarray:
    num_cols, num_gutters = len(colors), len(colors) - 1
    img_width = num_cols * col_width + num_gutters * gutter_width
    img_height = img_width // 2  # only render half of image, which will be reflected afterwards

    # Initialize image as all white
    data = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # For each column of color....
    for col_num in range(num_cols):
        xmin = col_num * (col_width + gutter_width)
        xmax = xmin + col_width

        # ...for each increasing thickness value...
        ymin = 0
        thickness = 1
        while ymin < img_height:
            # ...paint repetitions of stripes until there's no more image height left.
            for _ in range(num_repeats):
                ymax = min(ymin + thickness, img_height + 1)
                data[ymin:ymax, xmin:xmax] = colors[col_num]
                ymin += thickness * 2
                if ymin > img_height:
                    break
            if ymin > img_height or ymax > img_height:
                break
            thickness *= 2

    # Reflect array and mosaic output to be reflection, then 1 pixel white row, then original. This way the thinnest
    # stripes are in the middle of the image.
    data = np.vstack((np.flip(data, axis=0), np.full((1, img_width, 3), 255, np.uint8), data))

    # Add black border around the whole thing
    data = np.pad(
        data, ((gutter_width, gutter_width), (gutter_width, gutter_width), (0, 0)), mode='constant', constant_values=0
    )
    return data


if __name__ == '__main__':
    data = create_array_of_colored_stripes(COLORS, 64, 16)

    image = Image.fromarray(data)
    filename = 'image.png'
    image.save(filename)
    print(f'Saved data of size {data.shape} to {filename}')
