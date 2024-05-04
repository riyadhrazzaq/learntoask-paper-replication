from matplotlib import pyplot as plt
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore


def compute_metrics(hypotheses, references):
    bleu_n = []
    iterable_of_references = [[ref] for ref in references]
    for i in range(1, 5):
        bleu_n.append(BLEUScore(n_gram=i)(hypotheses, iterable_of_references).item())

    rouge_fn = ROUGEScore(rouge_keys="rougeL")
    rouge_score = rouge_fn(hypotheses, references)

    return {
        "bleu1": bleu_n[0],
        "bleu2": bleu_n[1],
        "bleu3": bleu_n[2],
        "bleu4": bleu_n[3],
        "rougeL": rouge_score["rougeL_fmeasure"].item(),
    }


def compute_metrics_from_file(hypothesis_file, reference_file):
    with open(hypothesis_file, "r") as f:
        hypotheses = f.readlines()
    with open(reference_file, "r") as f:
        references = f.readlines()

    return compute_metrics(hypotheses, references)


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
    fig.savefig("attention.png")