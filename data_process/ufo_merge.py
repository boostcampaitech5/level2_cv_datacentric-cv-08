import json
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "data_paths",
        nargs="+",
        help="UFO format json files to merge. Can be json files or directories contain json files.",
    )
    parser.add_argument("--target_path", type=str, default="../../data/New_sample/ufo")

    args = parser.parse_args()

    return args


def main(args):
    data_paths = args.data_paths
    save_path = args.target_path

    merged_ufo = {"images": {}}
    merged_dict_list = []

    filelist = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            files_in_dir = os.listdir(data_path)
            filelist += [os.path.join(data_path, file) for file in files_in_dir]
        else:
            filelist.append(data_path)

    for file in filelist:
        with open(file, "r") as f:
            json_obj = json.load(f)
        merged_dict_list += list(json_obj["images"].items())
    merged_ufo["images"] = dict(merged_dict_list)

    with open(os.path.join(save_path, "train.json"), "w") as f:
        json.dump(merged_ufo, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
