import os
import glob
import tqdm
import json
import numpy as np
from dataset import (
    filter_vertices,
    resize_img,
    adjust_height,
    rotate_img,
    crop_img,
    generate_roi_mask,
)
from east_dataset import generate_score_geo_maps
import albumentations as A
import pickle
from utils import read_json
from argparse import ArgumentParser
from PIL import Image


def preprocess(args):
    ignore_tags = args.ignore_tags
    ignore_under_threshold = 10
    drop_under_threshold = 1
    image_size = args.image_size
    crop_size = args.input_size

    morph = []
    transform = []
    if args.augmentation:
        morph = args.augmentation["morph"]

        funcs = []
        for t in args.augmentation["transform"]:
            funcs.append(__import__("augmentation").__dict__[t]())
        transform = A.Compose(funcs, p=args.augmentation["p_aug"])

    image_paths = glob.glob(os.path.join(args.image_dir, "*"))

    with open(args.json_path, "r") as f:
        anno = json.load(f)

    if not os.path.exists(args.processed_data_dir):
        os.makedirs(args.processed_data_dir)

    with open(os.path.join(args.processed_data_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    processed_data = {}
    print("Data Preprocessing.....")
    for image_fpath in tqdm.tqdm(image_paths):
        image_fname = os.path.basename(image_fpath)
        vertices, labels = [], []
        for word_info in anno["images"][image_fname]["words"].values():
            word_tags = word_info["tags"]

            ignore_sample = any(elem for elem in word_tags if elem in ignore_tags)
            num_pts = np.array(word_info["points"]).shape[0]

            # skip samples with ignore tag and
            # samples with number of points greater than 4
            if ignore_sample or num_pts > 4:
                continue

            vertices.append(np.array(word_info["points"]).flatten())
            labels.append(int(not word_info["illegibility"]))
        vertices, labels = np.array(vertices, dtype=np.float32), np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices, labels, ignore_under=ignore_under_threshold, drop_under=drop_under_threshold
        )
        image = Image.open(image_fpath)
        image, vertices = resize_img(image, vertices, image_size)
        image, vertices = adjust_height(image, vertices)
        # image, vertices = rotate_img(image, vertices)/
        image, vertices = crop_img(image, vertices, labels, crop_size)

        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)

        for m in morph:
            image = __import__("augmentation").__dict__[m](image)

        if transform:
            image = transform(image=image)["image"]

        image = A.Normalize(mean=(0.776, 0.772, 0.767), std=(0.218, 0.227, 0.240))(image=image)[
            "image"
        ]

        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)
        score_map, geo_map = generate_score_geo_maps(image, word_bboxes, map_scale=0.5)
        processed_data[image_fname] = {
            "roi_mask": roi_mask,
            "score_map": score_map,
            "geo_map": geo_map,
            "image": image,
        }

    with open(os.path.join(args.processed_data_dir, "processed_data.pickle"), "wb") as fw:
        pickle.dump(processed_data, fw)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="config file path (default: ./config.json)",
    )

    args = parser.parse_args()

    args = read_json(args.config)

    preprocess(args)
