import argparse

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
args.add_argument("experiment-name", type=str)

args.add_argument('--batch-size', type=int, default=cfg.batch_size)
args.add_argument('--lr', type=float, default=cfg.lr)
args.add_argument("--model-name", type=str, default=cfg.model_name)
args.add_argument("--max-step", type=int, default=-1)
args.add_argument("--max-length", type=int, default=cfg.max_length)
args.add_argument("--max-epoch", type=int, default=cfg.max_epoch)
args.add_argument("--no-pretrain", action="store_true")
args.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
args.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps)
args.add_argument("--nth-hidden-layer", type=int, default=cfg.nth_hidden_layer)

args = args.parse_args()

# build param dictionary from args
params = vars(args)
params = {k.replace('-', '_'): v for k, v in params.items()}


def main():
    train(params, disable_tqdm=True)


if __name__ == '__main__':
    main()
