import argparse
import logging

import torchtext

from config import load_config

torchtext.disable_torchtext_deprecation_warning()
import torch

from modelutil import generate, init_model
from metrics import compute_metrics
from trainutil import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# define arguments, override the defaults from config.py with arguments
args = argparse.ArgumentParser()
args.add_argument('srcfile', type=str)
args.add_argument('tgtfile', type=str)
args.add_argument('checkpoint_dir', type=str)
args.add_argument('--outfile', type=str, help="save generated output to this file")
args.add_argument('--search', choices=["greedy", "beam"], type=str, default="greedy", help="search method to use")

args = args.parse_args()

logger.info(f"Params: {vars(args)}")


def main():
    cfg = load_config(args.checkpoint_dir + "/history/config.yaml")

    src_tokenizer = torch.load(args.checkpoint_dir + "/src_tokenizer.pt")
    tgt_tokenizer = torch.load(args.checkpoint_dir + "/tgt_tokenizer.pt")
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab
    model = init_model(cfg, src_vocab, tgt_vocab)
    model, _, _, epoch = load_checkpoint(model, args.checkpoint_dir + "/model_best.pt")

    with open(args.srcfile) as srcfile:
        sources = srcfile.readlines()

    with open(args.tgtfile) as tgtfile:
        targets = tgtfile.readlines()

    hypotheses = []

    for i, source in enumerate(sources):
        hyp, _ = generate(model, source, src_tokenizer, tgt_tokenizer, cfg, method=args.search)
        hypotheses.append(hyp[0])

        if i % 100 == 0:
            logger.info(f"Generated {i}/{len(sources)} hypotheses")
        if i == 100:
            break

    metrics = compute_metrics(hypotheses, targets)
    metrics = {k: v * 100 for k, v in metrics.items()}
    print(metrics, len(hypotheses))
    hypotheses.extend(['','ami','hoi'])

    if args.outfile:
        with open(args.outfile, "w") as f:
            f.writelines([h + '\n' for h in hypotheses])


if __name__ == '__main__':
    main()
