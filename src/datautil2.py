import io
import sys
from pathlib import Path

sys.path.append('.')

from typing import List

import torch
import torchtext
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator


class TokenizerV2:
    def __init__(self, vocab: torchtext.vocab.Vocab):
        self.vocab = vocab
        self.pad_index = vocab['<PAD>']
        self.sos_index = vocab['<SOS>']
        self.eos_index = vocab['<EOS>']
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>']

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
            self, data_dir, srcfile, tgtfile, src_vocab, tgt_vocab
    ):
        """
        Given a directory and a prefix, load the source and target data

        Args:
            data_dir (Path): directory where the data is stored
            prefix (str): prefix of the data file, e.g. "train", "valid", "test"
            src_vocab (torchtext.vocab.Vocab): source vocabulary
            tgt_vocab (torchtext.vocab.Vocab): target vocabulary
            src_max_seq (int): maximum sequence length for source
            tgt_max_seq (int): maximum sequence length for target
        """
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        with open(data_dir / srcfile, "r") as f:
            self.src_lines = f.readlines()
        with open(data_dir / tgtfile, "r") as f:
            self.tgt_lines = f.readlines()
        
        self.src_tokenizer = TokenizerV2(src_vocab)
        self.tgt_tokenizer = TokenizerV2(tgt_vocab)


    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        return (
            self.src_lines[index],
            self.tgt_lines[index]
        )
    
    

    

class CollateFn:
    def __init__(self, src_tokenizer, tgt_tokenizer, src_max_seq=None, tgt_max_seq=None):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_max_seq=src_max_seq
        self.tgt_max_seq=tgt_max_seq
        
    def __call__(self, data: List[tuple]):
        src_max_seq = self.src_max_seq
        tgt_max_seq = self.tgt_max_seq
        sources = []
        targets = []
        
        for pair in data:
            sources.append(pair[0].strip())
            targets.append(pair[1].strip())
            
            if self.src_max_seq is None:
                num_tokens = len(pair[0].split())
                if num_tokens > src_max_seq:
                    src_max_seq = num_tokens
            
            if self.tgt_max_seq is None:
                num_tokens = len(pair[1].split())
                if num_tokens > tgt_max_seq:
                    tgt_max_seq = num_tokens

        
        src_tensor, src_mask = self.src_tokenizer.encode(sources, max_seq=src_max_seq+1, add_eos=False)

        tgt_tensor, tgt_mask = self.tgt_tokenizer.encode(targets, max_seq=tgt_max_seq+1, add_eos=True)
        
        return src_tensor, tgt_tensor, src_mask, tgt_mask
        