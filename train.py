import argparse

from datahandler import load_and_build_vocab, get_data_loader, load_or_build_models
from tokenization import Tokenizer


def main():
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
        "--hidden", type=int, default=300, help="Number of hidden units in the LSTM."
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
        "--unidirectioanl",
        action="store_true",
        help="Use a unidirectional LSTM in Encoder",
    )

    # Define training hyperparameters arguments
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate.")
    parser.add_argument(
        "--clip_norm", type=float, default=5.0, help="Gradient clipping norm."
    )
    parser.add_argument("--max_epoch", type=int, default=15, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

    # Define checkpoint and Neptune-related arguments
    parser.add_argument(
        "--existing_checkpoint",
        type=str,
        help="Path to checkpoint for resuming training.",
    )
    parser.add_argument(
        "--enable_neptune", action="store_true", help="Enable Neptune Cloud logging."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Placeholder for further code using the parsed arguments
    print(args)  # Example usage: print arguments for clarity
    run(args)


def prepare_config(args):
    config = {
        "hidden": args.hidden,
        "embedding": 300,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": not args.unidirectional,
        "lr": args.lr,
        "clip_norm": args.clip_norm,
        "max_epoch": args.max_epoch,
        "src_max_seq": args.src_max_seq,
        "tgt_max_seq": args.tgt_max_seq,
        "train_glove": args.train_glove,
        "batch_size": args.batch_size,
    }

    return config


def run(args):
    config = prepare_config(args)
    vocab = load_and_build_vocab(
        args.train_src_file, args.train_tgt_file, args.vocab_path
    )
    tokenizer = Tokenizer(vocab, vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"])
    train_dl = get_data_loader(
        args.train_src_file, args.train_tgt_file, tokenizer, config, True
    )

    valid_dl = get_data_loader(
        args.dev_src_file, args.dev_tgt_file, tokenizer, config, shuffle=False
    )

    model, optimizer, lr_scheduler, epoch = load_or_build_models(args, config, vocab)


if __name__ == "__main__":
    main()
