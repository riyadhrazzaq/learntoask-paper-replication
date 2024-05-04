import argparse

import torchtext

from config import validate_config, load_config

torchtext.disable_torchtext_deprecation_warning()

from trainutil import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# define arguments, override the defaults from config.py with arguments
args = argparse.ArgumentParser()
args.add_argument('training-file', type=str)
args.add_argument('validation-file', type=str)

args.add_argument("--config", type=str, default="config.yaml", help="path to the configuration file")

args = args.parse_args()


def _assert_dir_empty(cfg):
    # validate experiment directory is empty
    experiment_dir = cfg['checkpoint_dir'] + "/" + cfg["experiment_name"]
    if os.path.exists(experiment_dir):
        if len(os.listdir(experiment_dir)) > 0:
            raise ValueError(f"Experiment directory `{experiment_dir}` is not empty. Aborting to prevent "
                             f"overwriting.")


def main():
    cfg = load_config(args.config)
    _assert_dir_empty(cfg)

    train(cfg)


if __name__ == '__main__':
    main()
