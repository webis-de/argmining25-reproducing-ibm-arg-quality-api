import datetime
import torch
import random
import numpy as np


def set_seed(seed_val):
    torch.backends.cudnn.deterministic = True

    random.seed(seed_val)
    np.random.seed(seed_val)

    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_device():
    if torch.cuda.is_available():
        print('GPU(s) available. We will use:', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

