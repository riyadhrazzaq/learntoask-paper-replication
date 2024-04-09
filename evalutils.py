from pathlib import Path
from typing import List

import evaluate
import matplotlib
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights

device = "cuda" if torch.cuda.is_available() else "cpu"


def bleu_score(hypotheses: List[str], references: List[str]):
    """
    Args:
        hypotheses (List[str]): example: ['hello world', 'lorem ipsum sit dolor amet']
        references (List[str]): example: ^ same

    """
    assert len(hypotheses) == len(references)

    # adapting to bleu format, as we only have a single reference against a hypothesis
    list_of_references = [[ref] for ref in references]
    bleu = evaluate.load("bleu")
    results = bleu.compute(
        predictions=hypotheses, references=list_of_references, tokenizer=word_tokenize
    )
    return results["bleu"]


def generate(
    model,
    sentence,
    src_tokenizer,
    tgt_tokenizer,
    src_max_seq,
    tgt_max_seq,
    method="greedy",
):
    src_token_ids, src_mask = src_tokenizer.encode([sentence], max_seq=src_max_seq)
    src_token_ids = src_token_ids.to(device)
    src_mask = src_mask.to(device)
    if method == "greedy":
        tgt_token_ids, attention_scores = model.greedy_generate(
            src_token_ids, src_mask, tgt_max_seq, stop_at_eos=True
        )
    else:
        tgt_token_ids, prob = model.beam_generate(
            src_token_ids, src_mask, tgt_max_seq, stop_at_eos=True
        )

    tokens = tgt_tokenizer.decode(tgt_token_ids.view(1, -1), keep_specials=False)
    return tokens, attention_scores.squeeze(dim=3) if method == "greedy" else None


def plot_attention(attention, src, tgt):
    src = src.split()[: attention.size(0)]
    tgt = tgt.split()

    fig, ax = plt.subplots(figsize=(20, 8))
    heatmap = ax.pcolor(attention.T.cpu(), cmap=plt.cm.Blues, alpha=0.9)

    xticks = range(0, len(src))
    ax.set_xticks(xticks, minor=False)  # major ticks
    ax.set_xticklabels(src, minor=False, rotation=45)  # labels should be 'unicode'

    yticks = range(0, len(tgt))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(tgt, minor=False)  # labels should be 'unicode'

    ax.set_xlabel("source")
    ax.set_ylabel("target")

    ax.grid(False)

    # Save Figure
    plt.show()


def report_bleu(directory, prefix, model, src_tokenizer, tgt_tokenizer, out_dir=None):
    directory = Path(directory) if not isinstance(directory, Path) else directory
    hypotheses = []
    references = []
    with open(directory / f"{prefix}.src", "r") as src, open(
        directory / f"{prefix}.tgt"
    ) as tgt:
        for s, t in tqdm(zip(src, tgt)):
            s = s.strip()
            t = t.strip()

            hyp = generate(
                model,
                s,
                src_tokenizer,
                tgt_tokenizer,
                len(s.split()),
                len(t.split()),
                "beam",
            )
            hypotheses.append(hyp[0][0])
            references.append(t)

    if out_dir:
        with open(out_dir / f"{prefix}.hyp", "w") as f:
            for line in hypotheses:
                f.write(line + "\n")

    return bleu_score(hypotheses, references)
