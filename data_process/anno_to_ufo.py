import os
from argparse import ArgumentParser

from labelconverter import AIHubLabelConverter


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/New_sample/labels/",
        help="A json file or directory to convert annotation file to UFO format. If directory is given, converts every json file in given directory.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="../../data/New_sample/new_labels/",
        help="target directory to save converted annotation file.",
    )

    args = parser.parse_args()

    return args


def main(args):
    json_path = args.data_path
    target_dir = args.target_path

    if not os.path.isdir(json_path):
        json_path = [json_path]
    else:
        json_path = [json_path + json_file for json_file in os.listdir(json_path)]

    for json_file in json_path:
        label_converter = AIHubLabelConverter(json_file)
        label_converter.convert()
        label_converter.save_annotation(target_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
