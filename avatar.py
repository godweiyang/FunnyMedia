import argparse
import math
import os
import random
import PIL.Image as Image
from tqdm import tqdm


def generate(dir, size, rand):
    print(f"generating size {size}...")
    nums = len(os.listdir(dir))
    nums_width = int(math.sqrt(nums))
    nums_height = int((nums + nums_width - 1) / nums_width)
    img_width = nums_width * size
    img_height = nums_height * size

    image = Image.new("RGB", (img_width, img_height), "white")
    x = 0
    y = 0

    files = os.listdir(dir)
    if rand:
        random.shuffle(files)

    for i in tqdm(files):
        try:
            img = Image.open(os.path.join(dir, i))
        except IOError:
            print(i)
            print("image open error")
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
            image.paste(img, (x * size, y * size))
            x += 1
            if x == nums_width:
                x = 0
                y += 1
            img.close()

    image.save(f"avatar_{size}.jpg")
    image.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", "-d", type=str, default="avatar", help="directory of the avatars"
    )
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
        generate(args.dir, size, args.rand)
