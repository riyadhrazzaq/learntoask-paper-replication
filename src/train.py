import argparse
import torch
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import torchtext

try:
    torchtext.disable_torchtext_deprecation_warning()
except:
    pass

from config import load_config
from trainutil import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# define arguments, override the defaults from config.py with arguments
args = argparse.ArgumentParser()

args.add_argument("--config", type=str, default="config.yaml", help="path to the configuration file")
args.add_argument("--force", action="store_true", help="path to the configuration file")


args = args.parse_args()


def _assert_dir_empty(cfg):
    # validate experiment directory is empty
    experiment_dir = cfg['checkpoint_dir'] + "/" + cfg["experiment_name"]
    if os.path.exists(experiment_dir):
        if len(os.listdir(experiment_dir)) > 0:
            if args.force:
                logger.warning(f"Experiment directory `{experiment_dir}` is not empty. Proceeding anyway.")
            else:
                raise ValueError(f"Experiment directory `{experiment_dir}` is not empty. Aborting to prevent "
                                 f"overwriting.")


def main():
    cfg = load_config(args.config)
    _assert_dir_empty(cfg)

    train(cfg)


if __name__ == '__main__':
    main()
