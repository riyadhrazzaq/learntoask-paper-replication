import argparse
import logging
from statistics import mean
from typing import List

import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from datahandler import load_and_build_vocab, get_data_loader, load_or_build_models
from tokenization import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process model evaluation arguments.")

    # Define data arguments
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "src_file",
        type=str,
        help="Path to the source file. For example, sentences.txt where each samples are in a separate line",
    )

    parser.add_argument(
        "tgt_file",
        type=str,
        help="Path to the target file. For example, question.txt where each samples are in a separate line",
    )

    parser.add_argument(
        "vocab_path", type=str, help="Path to an existing vocab.pt file."
    )

    # define data preprocess arguments
    parser.add_argument(
        "--src_max_seq",
        type=int,
        default=50,
        help="Maximum sequence length for source.",
    )
    parser.add_argument(
        "--tgt_max_seq",
        type=int,
        default=15,
        help="Maximum sequence length for target.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of LSTM layers."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability between LSTM layers",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        help="Use a unidirectional LSTM in Encoder",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=300,
        help="Number of hidden units in the LSTM.",
    )

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--beam_only", action="store_true", help="Use beam search only."
    )
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width.")
    parser.add_argument(
        "--save_output", action="store_true", help="Save output to file."
    )

    parser.add_argument(
        "--train_glove", action="store_true", help="Train the GloVe embedding further."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Placeholder for further code using the parsed arguments
    run(args)


def run(args):
    config = {
        "lr": 1.0,
        "train_glove": args.train_glove,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": 300,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": not args.unidirectional,
        "batch_size": args.batch_size,
        "src_max_seq": args.src_max_seq,
        "tgt_max_seq": args.tgt_max_seq,
    }
    # prepare vocab and tokenizer
    logger.info("Loading vocab and tokenizer")
    vocab = load_and_build_vocab(None, None, args.vocab_path)
    tokenizer = Tokenizer(vocab, vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"])

    logger.info("Preparing data loaders")
    # prepare data loaders
    test_dl = get_data_loader(
        args.src_file, args.tgt_file, tokenizer, config, shuffle=False
    )

    with open(args.src_file) as f:
        src_test = [line.strip() for line in f]

    with open(args.tgt_file) as f:
        tgt_test = [line.strip() for line in f]
    logger.info(f"{len(src_test)} samples loaded for evaluation.")

    # prepare model
    logger.info("Loading model")
    model, optimizer, lr_scheduler, epoch = load_or_build_models(args, config, vocab)

    # now train
    logger.info("Beginning evaluation")
    if not args.beam_only:
        greedy_outputs, greedy_bleu = greedy(
            model, test_dl, config, tgt_test, tokenizer
        )
        print(f"Greedy BLEU: {greedy_bleu}")
    beam_outputs, beam_bleu = beam(
        model, src_test, tgt_test, config, args.beam_width, tokenizer
    )
    print(f"Beam BLEU: {beam_bleu}")

    if args.save_output:
        if not args.beam_only:
            with open("greedy_outputs.txt", "w") as f:
                for line in greedy_outputs:
                    f.write(line + "\n")
        with open("beam_outputs.txt", "w") as f:
            for line in beam_outputs:
                f.write(line + "\n")


def greedy(model, test_dl, config, target, tokenizer):
    outputs = []
    bleus = []
    i = 0
    for src_tensor in tqdm(test_dl):
        # when src_dl is built on SentenceQuestionDataset
        if isinstance(src_tensor, list):
            src_tensor = src_tensor[0]
        if target:
            batch_target = target[i : i + config["batch_size"]]

        src_tensor = src_tensor.to(device)

        token_ids = model.generate_batch(src_tensor)

        decoded_target = tokenizer.decode(token_ids, keep_specials=False)
        outputs.extend(decoded_target)

        if target:
            bleus.append(corpus_bleu(decoded_target, batch_target))

        i += config["batch_size"]

    return outputs, mean(bleus)


def beam(model, src: List[str], tgt: List[str], config, k, tokenizer):
    outputs = []
    bleus = []
    for s, q in zip(src, tgt):
        src_tensor = tokenizer.encode([s], max_seq=config["src_max_seq"])[0].view(1, -1)
        src_tensor = src_tensor.to(device)

        token_ids = model.beam_generate(src_tensor, k=k, max_seq=config["tgt_max_seq"])[
            0
        ]
        decoded_target = tokenizer.decode(token_ids, keep_specials=False)

        outputs.append(decoded_target)
        bleus.append(corpus_bleu(decoded_target, [q]))

    return outputs, mean(bleus)


if __name__ == "__main__":
    main()
