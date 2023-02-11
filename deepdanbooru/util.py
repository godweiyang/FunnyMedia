import numpy as np
from PIL import Image
import torch

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def tag(img, model):
    pic = resize_image(img, 512, 512)
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

    with torch.no_grad(), torch.autocast("cuda"):
        x = torch.from_numpy(a).to("cuda:0")
        y = model(x)[0].detach().cpu().numpy()

    probability_dict = {}

    for tag, probability in zip(model.tags, y):
        if probability < 0.8:
            continue
        if tag.startswith("rating:"):
            continue
        probability_dict[tag] = probability

    tags = sorted(probability_dict)
    res = []

    for tag in tags:
        tag_outformat = tag
        tag_outformat = (
            tag_outformat.replace("_", " ").replace("(", "").replace(")", "")
        )
        res.append(tag_outformat)

    return ", ".join(res)


def resize_image(im, width, height):
    def resize(im, w, h):
        if im.mode == "L":
            return im.resize((w, h), resample=LANCZOS)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    ratio = width / height
    src_ratio = im.width / im.height

    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width

    resized = resize(im, src_w, src_h)
    res = Image.new("RGB", (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        res.paste(
            resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0)
        )
        res.paste(
            resized.resize(
                (width, fill_height), box=(0, resized.height, width, resized.height)
            ),
            box=(0, fill_height + src_h),
        )
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        res.paste(
            resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0)
        )
        res.paste(
            resized.resize(
                (fill_width, height), box=(resized.width, 0, resized.width, height)
            ),
            box=(fill_width + src_w, 0),
        )

    return res
