import os
import random
import json

import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_output_dirs(out_dir):
    subdirs = ['checkpoints', 'logs', 'explanations',
               'tables', 'ablations', 'baselines', 'splits']
    for subdir in subdirs:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)


def save_config(config, out_dir):
    path = os.path.join(out_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def get_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'

