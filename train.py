import argparse


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process model training arguments.")

    # Define data arguments
    parser.add_argument(
        "source_file",
        type=str,
        help="Path to the source file. For example, sentences.txt where each samples are in a separate line",
    )
    parser.add_argument(
        "target_file",
        type=str,
        help="Path to the target file. For example, question.txt where each samples are in a separate line",
    )
    parser.add_argument(
        "--embedding_vector_path",
        type=str,
        help="Path to the embedding_vector.pt file.",
        required=False,
    )
    parser.add_argument(
        "--glove_embedding_path",
        type=str,
        help="Path to the GloVe embedding .txt file.",
        required=False,
    )

    # Define LSTM hyperparameters arguments
    parser.add_argument(
        "--hidden", type=int, default=300, help="Number of hidden units in the LSTM."
    )
    parser.add_argument(
        "--embedding", type=int, default=300, help="Dimension of the word embeddings."
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

    # Define checkpoint and Neptune-related arguments
    parser.add_argument(
        "--existing_checkpoint",
        type=str,
        help="Path to checkpoint for resuming training.",
    )
    parser.add_argument(
        "--enable_neptuna", action="store_true", help="Enable Neptuna Cloud logging."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Placeholder for further code using the parsed arguments
    print(args)  # Example usage: print arguments for clarity

    model, optimizer, lr_scheduler = load_or_new(args)


if __name__ == "__main__":
    main()
