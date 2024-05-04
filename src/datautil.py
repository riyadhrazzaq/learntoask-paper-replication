import io
import sys
from pathlib import Path

sys.path.append('.')

from typing import List

import torch
import torchtext
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    def __init__(self, vocab: torchtext.vocab.Vocab, pad_index, sos_index, eos_index):
        self.vocab = vocab
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.special_tokens = vocab.lookup_tokens(
            [self.pad_index, self.sos_index, self.eos_index]
        )

    def encode(self, text: List[str],
               add_sos=False, add_eos=False, max_seq=None) -> (torch.Tensor, torch.Tensor):
        """
        given a list of texts, return a tensor of token indices and another tensor of mask
        Args:
            text (List[str]): list of texts
            add_sos (bool): whether to add <sos> token at the beginning
            add_eos (bool): whether to add <eos> token at the end
            max_seq (int): maximum sequence length, if None, it will be the maximum length of the text

        Returns:
            (torch.Tensor, torch.Tensor): tensor of token indices, tensor of mask
        """
        batch_size = len(text)
        text_indexed = [self.vocab(document.split()) for document in text]
        text_lengths = [len(tokens) for tokens in text_indexed]
        max_seq = max_seq if max_seq else max(text_lengths)

        text_tensor = torch.full(
            (batch_size, max_seq), self.pad_index, dtype=torch.long
        )

        for i, data in enumerate(text_indexed):
            # clip sentence length to max_seq or less to keep space for <sos>, <eos>

            token_ids = (
                data
                if text_lengths[i] <= (max_seq - int(add_sos) - int(add_eos))
                else data[: (max_seq - int(add_sos) - int(add_eos))]
            )

            if add_sos:
                token_ids = [
                                self.sos_index,
                            ] + token_ids

            if add_eos:
                token_ids = token_ids + [
                    self.eos_index,
                ]
            # recalculate because it might be less than max_seq
            text_tensor[i, : len(token_ids)] = torch.tensor(token_ids)

        mask = text_tensor != self.pad_index
        return text_tensor, mask

    def decode(self, token_ids: torch.Tensor, keep_specials=True) -> List[str]:
        """
        Given a tensor of token indices, return a list of texts

        Args:
            token_ids (torch.Tensor): required shape: (N, sequence)
            keep_specials (bool): whether to keep special tokens or not in the output

        Returns:
            List[str]: list of texts
        """
        text = []
        for token_id in token_ids:
            tokens = self.vocab.lookup_tokens(token_id.tolist())
            if not keep_specials:
                tokens = [token for token in tokens if token not in self.special_tokens]

            text.append(" ".join(tokens))

        return text


def yield_token(text_file_path):
    with io.open(text_file_path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def load_and_build_vocab(sentence_path, question_path, src_vocab_size, tgt_vocab_size):
    sentence_vocab = build_vocab_from_iterator(
        yield_token(sentence_path),
        max_tokens=src_vocab_size,
        specials=["<SOS>", "<EOS>", "<PAD>", "<UNK>"],
        special_first=True,
    )
    sentence_vocab.set_default_index(sentence_vocab["<UNK>"])

    question_vocab = build_vocab_from_iterator(
        yield_token(question_path),
        max_tokens=tgt_vocab_size,
        specials=["<SOS>", "<EOS>", "<PAD>", "<UNK>"],
        special_first=True,
    )
    question_vocab.set_default_index(question_vocab["<UNK>"])

    return sentence_vocab, question_vocab


class SourceTargetDataset(Dataset):
    def __init__(
            self, srcfile, tgtfile, src_vocab, tgt_vocab, src_max_seq, tgt_max_seq,
            return_tokenizers=False
    ):
        """
        Given filepaths, load the source and target data and build tensors

        Args:
            prefix (str): prefix of the data file, e.g. "train", "valid", "test"
            src_vocab (torchtext.vocab.Vocab): source vocabulary
            tgt_vocab (torchtext.vocab.Vocab): target vocabulary
            src_max_seq (int): maximum sequence length for source
            tgt_max_seq (int): maximum sequence length for target
        """
        super().__init__()

        with open(srcfile, "r") as f:
            src_lines = f.readlines()
        with open(tgtfile, "r") as f:
            tgt_lines = f.readlines()

        self.src_tensor, self.src_mask, self.src_tokenizer = self._build_tensor(
            src_lines, src_vocab, src_max_seq
        )
        self.tgt_tensor, self.tgt_mask, self.tgt_tokenizer = self._build_tensor(
            tgt_lines, tgt_vocab, tgt_max_seq, add_eos=True
        )

        assert self.src_tensor.size(0) == self.tgt_tensor.size(0)

        if not return_tokenizers:
            del self.src_tokenizer, self.tgt_tokenizer


    def _build_tensor(
            self, lines: List[str], vocab: torchtext.vocab.Vocab, max_seq, add_eos=False
    ):
        """helper function to build tensor and mask from lines of text"""

        tokenizer = Tokenizer(vocab, vocab["<PAD>"], vocab["<SOS>"], vocab["<EOS>"])
        tensor, mask = tokenizer.encode(lines, max_seq=max_seq, add_eos=add_eos)

        return tensor, mask, tokenizer

    def __len__(self):
        return self.src_tensor.size(0)

    def __getitem__(self, index):
        return (
            self.src_tensor[index],
            self.tgt_tensor[index],
            self.src_mask[index],
            self.tgt_mask[index],
        )

