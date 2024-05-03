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

