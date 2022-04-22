import argparse
import os
import random
import PIL.Image as Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm, trange


def mean_pixel(colors):
    colors = [0.3 * r + 0.59 * g + 0.11 * b for r, g, b in colors]
    return int(np.mean(colors))


def generate(dir, source, size, rand):
    print(f"generating size {size}...")

    files = os.listdir(dir)
    if rand:
        random.shuffle(files)

    image = Image.open(source)
    image = image.convert("RGB")
    img_width, img_height = image.size
    img_width = ((img_width + size - 1) // size) * size * ((size + 9) // 10)
    img_height = ((img_height + size - 1) // size) * size * ((size + 9) // 10)
    image = image.resize((img_width, img_height), Image.ANTIALIAS)

    colors = defaultdict(list)
    for i in tqdm(files):
        try:
            img = Image.open(os.path.join(dir, i))
        except IOError:
            print(i)
            print("image open error")
        else:
            img = img.convert("RGB")
            img = img.resize((size, size), Image.ANTIALIAS)
            colors[mean_pixel(img.getdata())].append(i)
            img.close()
    for i in range(256):
        if len(colors[i]) == 0:
            for n in range(1, 256):
                if len(colors[i - n]) != 0:
                    colors[i] = colors[i - n]
                    break
                if len(colors[i + n]) != 0:
                    colors[i] = colors[i + n]
                    break

    index = defaultdict(int)
    for i in trange(0, img_width, size):
        for j in range(0, img_height, size):
            now_colors = []
            for ii in range(i, i + size):
                for jj in range(j, j + size):
                    now_colors.append(image.getpixel((ii, jj)))
            mean_color = mean_pixel(now_colors)
            img = Image.open(
                os.path.join(
                    dir, colors[mean_color][index[mean_color] % len(colors[mean_color])]
                )
            )
            img = img.convert("RGB")
            img = img.resize((size, size), Image.ANTIALIAS)
            image.paste(img, (i, j))
            img.close()
            index[mean_color] += 1

    source_name = ".".join(source.split(".")[:-1])
    image.save(f"{source_name}_{size}.jpg")
    image.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", "-d", type=str, default="avatar", help="directory of the avatars"
    )
    parser.add_argument("--img", "-i", type=str, help="source image to be coverd")
    parser.add_argument(
        "--size",
        "-s",
        type=str,
        default="30",
        help="size of each avatar (size1,size2,...)",
    )
    parser.add_argument(
        "--rand",
        "-r",
        action="store_true",
        help="whether to shuffle the avatars",
    )
    args = parser.parse_args()
    sizes = [int(s) for s in args.size.split(",")]
    for size in sizes:
        generate(args.dir, args.img, size, args.rand)
