import os

import yaml


def load_config(path):
    with open(path, "r") as stream:
        cfg = yaml.safe_load(stream)
        validate_config(cfg)
    return cfg


def validate_config(cfg):
    # strip trailing slashes in directory strings, leads to confusion
    for k, v in cfg.items():
        if "_dir" in k and v[-1] == "/":
            cfg[k] = v[:-1]
