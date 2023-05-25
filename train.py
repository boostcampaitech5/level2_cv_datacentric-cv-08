import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from utils import read_json, seed_everything
import wandb

def parse_args():
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

    seed_everything(args.seed)

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def do_training(
                data_dir,
                model_dir,
                device,
                image_size,
                input_size,
                num_workers,
                batch_size,
                learning_rate,
                max_epoch,
                save_interval,
                ignore_tags,
                seed,
                extractor_pth,
                enable_amp,
                project_name,
                val_ratio,
                val_batch_size,
                patience_limit,
                num_accumulation_step                
                ):
    assert num_accumulation_step >= 0, "Gradient Accumulation step must be >= 0"
    dataset = SceneTextDataset(
        data_dir,
        split="train",
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        color_jitter=False,
    )
    dataset = EASTDataset(dataset)
    train_size = int(len(dataset) * (1 - val_ratio))
    valid_size = len(dataset) - train_size
    trainset, validset = random_split(dataset, [train_size, valid_size])

    train_num_batches = math.ceil(len(trainset) / batch_size)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    if val_ratio > 0:
        valid_num_batches = math.ceil(len(validset) / val_batch_size)
        valid_loader = DataLoader(
                validset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
            )    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(extractor_pth=extractor_pth)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    if enable_amp:
        scaler = torch.cuda.amp.GradScaler()
        
    # Early Stop
    best_mean_loss = float('inf')
    patience_limit = patience_limit
    patience = 0

    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=train_num_batches) as pbar:
            for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                if enable_amp:
                    # Mixed Precision으로 Operation들을 casting
                    with torch.cuda.amp.autocast(enable_amp):
                        loss, extra_info = model.train_step(
                            img, gt_score_map, gt_geo_map, roi_mask
                        )
                    if num_accumulation_step == 0:
                        optimizer.zero_grad()
                        # Loss를 scaling한 후에 backward진행
                        scaler.scale(loss).backward()
                        # 원래 scale에 맞추어 gradient를 unscale하고 optimizer를 통한 gradient update
                        scaler.step(optimizer)
                        # 다음 iter를 위한 scale update
                        scaler.update()
                    if num_accumulation_step > 0:
                        scaler.scale(loss).backward()
                        if ((idx + 1) % num_accumulation_step == 0) or (idx + 1 == len(train_loader)):
                            scaler.step(optimizer)
                            # 다음 iter를 위한 scale update
                            scaler.update()                          
                            optimizer.zero_grad()


                else:
                    loss, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )
                    if num_accumulation_step == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    if num_accumulation_step > 0:
                        loss = loss / num_accumulation_step
                        loss.backward()
                        if ((idx + 1) % num_accumulation_step == 0) or (idx + 1 == len(train_loader)):
                            optimizer.step()
                            optimizer.zero_grad()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(val_dict)

                wandb.log(
                    {
                        "train/dice_score": 1 - extra_info["cls_loss"],
                        "train/epoch_loss": epoch_loss,
                        "train/cls_loss": extra_info["cls_loss"],
                        "train/angle_loss": extra_info["angle_loss"],
                        "train/iou_loss": extra_info["iou_loss"],
                    }
                )

        scheduler.step()

        mean_loss = epoch_loss / train_num_batches
        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                mean_loss, timedelta(seconds=time.time() - epoch_start)
            )
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/mean_loss": mean_loss,
            }
        )

        if val_ratio > 0:
            valid_infos = {
                "val/loss" : 0.0,
                "val/dice_score" : 0.0,
                "val/cls_loss" : 0.0,
                "val/angle_loss" : 0.0,
                "val/iou_loss" : 0.0
            }
            valid_start = time.time()
            with tqdm(total=valid_num_batches) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                    pbar.set_description("[Valid]")
                    with torch.no_grad():
                        model.eval()
                        loss, extra_info = model.train_step(
                            img, gt_score_map, gt_geo_map, roi_mask
                        )
                        valid_infos["val/loss"] += loss.item() / valid_num_batches
                        valid_infos["val/cls_loss"] += extra_info["cls_loss"] / valid_num_batches
                        valid_infos["val/angle_loss"] += extra_info["angle_loss"] / valid_num_batches
                        valid_infos["val/iou_loss"] += extra_info["iou_loss"] / valid_num_batches
                        valid_infos["val/dice_score"] += ((1 - extra_info["cls_loss"]) / valid_num_batches)
                        pbar.update(1)
                        val_dict = {
                            "Valid Cls loss": extra_info["cls_loss"],
                            "Valid Angle loss": extra_info["angle_loss"],
                            "Valid IoU loss": extra_info["iou_loss"],
                        }
                        pbar.set_postfix(val_dict)                        
            print(
                "[Valid] valid loss: {:.4f} | Elapsed time: {}".format(
                    valid_infos["val/loss"], timedelta(seconds=time.time() - valid_start)
                )
            )
            wandb.log(
                {
                    "epoch": epoch,
                    **valid_infos
                }
            )        

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)
        
        # Early Stop
        if best_mean_loss > mean_loss:
            best_mean_loss = mean_loss
            patience = 0
            
            os.makedirs(model_dir, exist_ok=True)
            ckpt_fpath = osp.join(model_dir, 'best.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        else:
            patience += 1
            if patience >= patience_limit:
                break
    
    print(f"Best Mean Loss : {best_mean_loss}")


def main(args):
    wandb.init(
        entity="level2-hiboostcamp-2",
        project="data-centric",
        name=args.project_name,
        config=args,
    )

    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)
