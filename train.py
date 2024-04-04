import argparse
import logging

from datahandler import load_and_build_vocab, get_data_loader, load_or_build_models
from tokenization import Tokenizer
from utils import fit

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def get_argparse(hidden_dim=300, src_max_seq=40, tgt_max_seq=15):
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process model training arguments.")

    # Define data arguments
    parser.add_argument(
        "train_src_file",
        type=str,
        help="Path to the source file. For example, sentences.txt where each samples are in a separate line",
    )

    parser.add_argument(
        "train_tgt_file",
        type=str,
        help="Path to the target file. For example, question.txt where each samples are in a separate line",
    )

    parser.add_argument(
        "dev_src_file",
        type=str,
        help="Path to the validation source file. For example, sentences.txt where each samples are in a separate line",
    )
    parser.add_argument(
        "dev_tgt_file",
        type=str,
        help="Path to the validation target file. For example, question.txt where each samples are in a separate line",
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
    parser.add_argument(
        "--vocab_path", type=str, help="Path to an existing vocab.pt file."
    )

    # define data preprocess arguments
    parser.add_argument(
        "--src_max_seq",
        type=int,
        default=src_max_seq,
        help="Maximum sequence length for source.",
    )
    parser.add_argument(
        "--tgt_max_seq",
        type=int,
        default=tgt_max_seq,
        help="Maximum sequence length for target.",
    )

    # Define LSTM hyperparameters arguments
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=hidden_dim,
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
    return parser.parse_args()


def main():
    args = get_argparse(hidden_dim=300, src_max_seq=40, tgt_max_seq=15)
    # Placeholder for further code using the parsed arguments
    print(args)  # Example usage: print arguments for clarity
    run(args)


def prepare_config(args):
    config = {
        "hidden_dim": args.hidden_dim,
        "embedding_dim": 300,
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

    return config


def run(args):
    config = prepare_config(args)

    # prepare vocab and tokenizer
    logger.info("Preparing vocab and tokenizer")
    vocab = load_and_build_vocab(
        args.train_src_file, args.train_tgt_file, args.vocab_path
    )
    tokenizer = Tokenizer(vocab, vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"])

    logger.info("Preparing training and validation data loaders")
    # prepare data loaders
    train_dl = get_data_loader(
        args.train_src_file, args.train_tgt_file, tokenizer, config, True
    )
    valid_dl = get_data_loader(
        args.dev_src_file, args.dev_tgt_file, tokenizer, config, shuffle=False
    )

    # prepare model
    logger.info("Preparing model and optimizer")
    model, optimizer, lr_scheduler, epoch = load_or_build_models(
        args.checkpoint,
        args.embedding_vector_path,
        args.glove_embedding_dir,
        config,
        vocab,
    )

    # now train
    logger.info("Beginning training")
    fit(
        model,
        optimizer,
        train_dl,
        valid_dl,
        config,
        args,
        lr_scheduler,
        max_step=config["max_step"],
        experiment_name=args.experiment_name,
        epoch=epoch,
    )


if __name__ == "__main__":
    main()
