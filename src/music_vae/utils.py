import numpy as np
import importlib

import torch

from sacred.observers import FileStorageObserver


def get_class_by_name(cls):
    if cls is None:
        return None
    module_name, class_name = cls.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def load_class_by_name(cls, *args, **kwargs):
    if cls is None:
        return None
    module_name, class_name = cls.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)(*args, **kwargs)


def get_run_dir(_run):
    for obs in _run.observers:
        if isinstance(obs, FileStorageObserver):
            return obs.dir
    return None


def save_ckpt(file_path, global_step, model, optimizer, scheduler):
    """Save current training state"""
    ckpt = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "model_kwargs": model.create_ckpt(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(ckpt, file_path)


def load_ckpt(ckpt_path, model, opt, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # load model state
    model.load_state_dict(ckpt["model_state_dict"])

    # load optimizer state
    opt.load_state_dict(ckpt["optimizer"])

    # load scheduler path
    scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt["global_step"]


def get_grad_norm(params):
    # alternative https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2
    # total_norm = 0.0
    # for p in params:
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    # return total_norm ** (1. / 2)
    # see https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
    norm_type = 2
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
    return total_norm.item()


def get_beta_fn(start_beta, end_beta, duration):
    beta_rate = np.exp(np.log(1.0 + start_beta - end_beta) / duration)
    return lambda step: np.minimum((1 + start_beta - beta_rate**step), end_beta)


def get_sampling_fn(rate, schedule):
    if schedule == "constant":
        return lambda _: rate

    return lambda step: 1.0 - rate / (rate + np.exp(step / rate))


