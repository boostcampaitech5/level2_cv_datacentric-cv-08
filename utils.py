from pathlib import Path
from easydict import EasyDict
import json
import torch
import numpy as np
import random
import os


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=EasyDict)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(seed):
    def func(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return func
