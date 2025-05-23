import torch
import gc

# 存放跨平台的工具类


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
