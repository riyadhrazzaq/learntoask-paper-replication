import argparse
import logging
import pprint
import random

import numpy as np
import torchtext

try:
    torchtext.disable_torchtext_deprecation_warning()
except:
    pass

from config import load_config

import torch

from modelutil import generate, init_model
from metrics import compute_metrics
from trainutil import load_checkpoint

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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
args.add_argument('--search', choices=["greedy", "beam", "nucleus"], type=str, default="greedy",
                  help="search method to use")
args.add_argument('--p', type=float, default=0.9, help="nucleus sampling p value")

args = args.parse_args()

logger.info(f"Params: {vars(args)}")


def main():
    cfg = load_config(args.checkpoint_dir + "/history/config.yaml")

    src_tokenizer = torch.load(args.checkpoint_dir + "/src_tokenizer.pt")
    tgt_tokenizer = torch.load(args.checkpoint_dir + "/tgt_tokenizer.pt")
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab

    # during generation, loading pretrained glove embedding from file is not required
    # as they are already in the checkpoint
    model = init_model(cfg, src_vocab, tgt_vocab)
    model, _, _, epoch = load_checkpoint(model, args.checkpoint_dir + "/model_best.pt")

    with open(args.srcfile) as srcfile:
        sources = srcfile.readlines()

    with open(args.tgtfile) as tgtfile:
        targets = tgtfile.readlines()

    hypotheses = generate_hypotheses(sources, model, src_tokenizer, tgt_tokenizer, cfg, args)
    metrics = compute_metrics(hypotheses, targets)
    metrics = {k: v * 100 for k, v in metrics.items()}
    pprint.pp(metrics)

    if args.outfile:
        with open(args.outfile, "w") as f:
            f.writelines([h + '\n' for h in hypotheses])


def generate_hypotheses(sources, model, src_tokenizer, tgt_tokenizer, cfg, args):
    hypotheses = []
    # generate samplewise
    if args.search in ["beam", "greedy"]:
        for i, source in enumerate(sources):
            hyp, _ = generate(model, source, src_tokenizer, tgt_tokenizer, cfg, method=args.search)
            hypotheses.append(hyp[0])

            if i % 100 == 0:
                print(f"Generated {i}/{len(sources)} hypotheses", end='\r')

    # generate batchwise
    elif args.search == "nucleus":
        assert args.p is not None and 1.0 > args.p > 0.0, "p must be in (0, 1)"
        batch_size = 64
        i = 0
        while i < len(sources):
            hyp, _ = generate(model, sources[i: i + batch_size], src_tokenizer, tgt_tokenizer, cfg, method="nucleus",
                              p=args.p)
            hypotheses.extend(hyp)
            i += batch_size
            print(f"Generated {i}/{len(sources)} hypotheses", end='\r')

    return hypotheses


if __name__ == '__main__':
    main()
