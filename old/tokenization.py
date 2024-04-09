from typing import List, Union
import torchtext
import torch


class Tokenizer:
    def __init__(self, vocab: torchtext.vocab.Vocab, pad_index, sos_index, eos_index):
        self.vocab = vocab
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.special_tokens = vocab.lookup_tokens(
            [self.pad_index, self.sos_index, self.eos_index]
        )

    def encode(self, text: List[str], add_sos=False, add_eos=False, max_seq=None):
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

    def decode(self, token_ids: torch.Tensor, keep_specials=True):
        text = []
        for token_id in token_ids:
            tokens = self.vocab.lookup_tokens(token_id.tolist())
            if not keep_specials:
                tokens = [token for token in tokens if token not in self.special_tokens]

            text.append(" ".join(tokens))

        return text
