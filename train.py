import argparse
import logging
from pathlib import Path

import torch
import torchtext
from torch.utils.data import DataLoader

from datautils import load_and_build_vocab, Tokenizer, SrcTgtDataset
from evalutils import report_bleu
from models import Seq2Seq
from trainutils import fit

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process model training arguments.")

    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where train.src, train.tgt, dev.src, dev.tgt exists.",
    )
    parser.add_argument(
        "--embedding_vector_path",
        type=str,
        help="Path to the embedding_vector.pt file.",
        required=False,
    )
    parser.add_argument(
        "--glove_embedding_dir",
        type=str,
        help="Directory to use for GloVe embeddings. Default is .vector_cache/",
        required=False,
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

    # Define LSTM hyperparameters arguments
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="Number of hidden units in the LSTM.",
    )

    parser.add_argument(
        "--train_glove", action="store_true", help="Train the GloVe embedding further."
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

    # Define training hyperparameters arguments
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate.")
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="Learning rate decay."
    )
    parser.add_argument(
        "--lr_decay_from", type=int, default=8, help="Learning rate decay from epoch."
    )
    parser.add_argument(
        "--clip_norm", type=float, default=5.0, help="Gradient clipping norm."
    )
    parser.add_argument("--max_epoch", type=int, default=15, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

    parser.add_argument(
        "--max_step", type=int, default=float("inf"), help="Number of steps per epoch"
    )

    # Define checkpoint and Neptune-related arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for resuming training.",
    )
    parser.add_argument(
        "--enable_neptune", action="store_true", help="Enable Neptune Cloud logging."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        help="meaningful experiment name for search",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Placeholder for further code using the parsed arguments
    print(args)  # Example usage: print arguments for clarity
    run(args)


def select_params_from_args(args):
    params = {
        "hidden_size": args.hidden_size,
        "embedding_size": 300,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": not args.unidirectional,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "lr_decay_from": args.lr_decay_from,
        "clip_norm": args.clip_norm,
        "max_epoch": args.max_epoch,
        "max_step": args.max_step,
        "src_max_seq": args.src_max_seq,
        "tgt_max_seq": args.tgt_max_seq,
        "train_glove": args.train_glove,
        "batch_size": args.batch_size,
    }

    return params


def run(args):
    params = select_params_from_args(args)
    data_dir = Path(args.directory)

    # prepare vocab and tokenizer
    logger.info("Preparing vocab and tokenizer")
    src_vocab, tgt_vocab = load_and_build_vocab(
        data_dir / "train.src", data_dir / "train.tgt"
    )

    train_ds = SrcTgtDataset(
        data_dir,
        "train",
        src_vocab,
        tgt_vocab,
        src_max_seq=params["src_max_seq"],
        tgt_max_seq=params["tgt_max_seq"],
    )
    dev_ds = SrcTgtDataset(
        data_dir,
        "dev",
        src_vocab,
        tgt_vocab,
        src_max_seq=params["src_max_seq"],
        tgt_max_seq=params["tgt_max_seq"],
    )
    src_tokenizer = Tokenizer(
        src_vocab, src_vocab["<PAD>"], src_vocab["<SOS>"], src_vocab["<EOS>"]
    )
    tgt_tokenizer = Tokenizer(
        tgt_vocab, tgt_vocab["<PAD>"], tgt_vocab["<SOS>"], tgt_vocab["<EOS>"]
    )

    logger.info("Preparing training and validation data loaders")
    params["batch_size"] = 64
    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=params["batch_size"], shuffle=False)

    logger.info("loading vector into memory")
    glove_vec = torchtext.vocab.Vectors(
        name="glove.840B.300d.txt", cache=args.glove_embedding_dir
    )
    src_embedding_vector = glove_vec.get_vecs_by_tokens(src_vocab.get_itos())
    tgt_embedding_vector = glove_vec.get_vecs_by_tokens(tgt_vocab.get_itos())

    # prepare model
    logger.info("Preparing model and optimizer")
    net = Seq2Seq(
        len(src_vocab),
        len(tgt_vocab),
        src_embedding_vector,
        tgt_embedding_vector,
        tgt_vocab["<PAD>"],
        tgt_vocab["<SOS>"],
        tgt_vocab["<EOS>"],
        hidden_size=params["hidden_size"],
        bidirectional=True,
        num_layers=params["num_layers"],
        src_embedding_size=300,
        tgt_embedding_size=300,
    )

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params["lr"])
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda epoch: (
            params["lr_decay"] if epoch > params["lr_decay_from"] else 1.0
        ),
    )

    # now train
    logger.info("Beginning training")
    history = fit(
        model=net,
        optimizer=optimizer,
        train_dl=train_dl,
        valid_dl=dev_dl,
        params=params,
        max_epoch=params["max_epoch"],
        lr_scheduler=lr_scheduler,
        max_step=params["max_step"],
        experiment_name=args.experiment_name,
        enable_neptune=args.enable_neptune,
    )

    logger.info("Evaluating validation split")
    val_bleu = report_bleu(
        data_dir, "dev", net, src_tokenizer, tgt_tokenizer, out_dir=history.exp_dir
    )
    logger.info("val bleu Score %s", val_bleu)
    history.log("val/bleu", val_bleu)
    history.stop()


if __name__ == "__main__":
    main()
