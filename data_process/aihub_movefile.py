import json
import os
import shutil
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument("--data_dir", type=str, default="../../data/New_sample")

    args = parser.parse_args()

    return args


def main(args):
    data_path = args.data_dir

    label_path = os.path.join(data_path, "라벨링데이터/인.허가/5350109")
    image_path = os.path.join(data_path, "원천데이터/인.허가/5350109")
    year_path_list = ["1994", "1997", "1999", "2001"]

    target_label_path = os.path.join(data_path, "labels")
    target_image_path = os.path.join(data_path, "img")

    os.makedirs(target_label_path, exist_ok=True)
    os.makedirs(target_image_path, exist_ok=True)

    for year in year_path_list:
        filelist = os.listdir(os.path.join(label_path, year))
        for file in filelist:
            shutil.move(
                os.path.join(label_path, year, file),
                os.path.join(target_label_path, file),
            )

    for year in year_path_list:
        filelist = os.listdir(os.path.join(image_path, year))
        for file in filelist:
            shutil.move(
                os.path.join(image_path, year, file),
                os.path.join(target_image_path, file),
            )
    
    # 기존 데이터셋의 label, image불일치로 인한 오류 해결
    for file in os.listdir(target_image_path):
        file_original = file
        file = file.split("-")
        if len(file) == 4 and file[1] == "1999" and file[2] == "0002":
            file[2] = "0001"
            shutil.move(
                os.path.join(target_image_path, file_original),
                os.path.join(target_image_path, "-".join(file))
            )
        


if __name__ == "__main__":
    args = parse_args()
    main(args)
