import io
from pathlib import Path
from typing import Union

import torch
import torchtext
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torch.utils.data import DataLoader, Dataset

from dataset import SentenceQuestionDataset
from models import Seq2Seq
from tokenization import Tokenizer
from utils import load_checkpoint


def yield_token(text_file_path):
    with io.open(text_file_path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def load_and_build_vocab(sentence_path, question_path, vocab_path=None):
    if vocab_path:
        return torch.load(vocab_path)

    sentence_vocab = build_vocab_from_iterator(
        yield_token(sentence_path),
        max_tokens=45000,
        specials=["<SOS>", "<EOS>", "<PAD>", "<UNK>"],
        special_first=True,
    )

    question_vocab = build_vocab_from_iterator(
        yield_token(question_path), max_tokens=28000
    )

    # merge two vocabs once collected from separate corpus
    vocab = torchtext.vocab.Vocab(sentence_vocab)
    vocab.set_default_index(vocab["<UNK>"])

    for token in question_vocab.get_itos():
        if token not in vocab:
            vocab.append_token(token)

    torch.save(vocab, "./checkpoint/vocab.pt")

    return vocab


def load_pretrained_glove(vocab, cache=None):
    embedding_vector = torch.zeros(size=(len(vocab), 300))
    glove = GloVe(cache="data/")
    for index in range(len(vocab)):
        embedding_vector[index] = glove[vocab.lookup_token(index)]

    return embedding_vector


def get_data_loader(
    src_file: Union[str, Path],
    tgt_file: Union[str, Path],
    tokenizer: Tokenizer,
    config: dict,
    shuffle=False,
):
    with open(src_file) as f:
        src = [line.strip() for line in f]

    with open(tgt_file) as f:
        tgt = [line.strip() for line in f]

    src_tensor, src_mask = tokenizer.encode(src, max_seq=config["src_max_seq"])
    tgt_tensor, tgt_mask = tokenizer.encode(
        tgt, add_eos=True, max_seq=config["tgt_max_seq"]
    )

    dataset = SentenceQuestionDataset(src_tensor, tgt_tensor, src_mask, tgt_mask)

    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)


def load_or_build_models(args, config, vocab):
    # load embedding vector
    if args.checkpoint:
        # if a checkpoint exists, don't bother wasting time on glove
        # the checkpoint already has the valid embedding weights
        embedding_vector = torch.zeros(size=(len(vocab), config["embedding_dim"]))
    else:
        if args.embedding_vector_path and Path(args.embedding_vector_path).is_file():
            glove = torch.load(args.embedding_vector_path)
        else:
            glove = GloVe(
                name="840B", dim=config["embedding_dim"], cache=args.glove_embedding_dir
            )

        embedding_vector = torch.zeros(size=(len(vocab), config["embedding_dim"]))
        for index in range(len(vocab)):
            embedding_vector[index] = glove[vocab.lookup_token(index)]

    model = Seq2Seq(
        len(vocab),
        embedding_vector,
        config["embedding_dim"],
        vocab["<PAD>"],
        vocab["<SOS>"],
        vocab["<EOS>"],
        hidden_dim=config["hidden_dim"],
        bidirectional=config["bidirectional"],
        num_layers=config["num_layers"],
        train_glove=config["train_glove"],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.5 if epoch > 8 else 1.0
    )

    # load if already exists
    epoch = 0
    if args.checkpoint:
        model, optimizer, lr_scheduler, epoch = load_checkpoint(
            model, args.checkpoint, optimizer, lr_scheduler
        )

    return model, optimizer, lr_scheduler, epoch
