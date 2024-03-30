from typing import List, Union
import torchtext


class Tokenizer:
    def __init__(self, vocab: torchtext.vocab.Vocab, pad_index, sos_index, eos_index):
        self.vocab = vocab
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.special_tokens = [vocab.lookup_tokens([self.pad_index, self.sos_index, self.eos_index])]

    
    def encode(self, text: List[str], add_sos=False, add_eos=False, max_seq=None):
        batch_size = len(text)
        text_lengths = [len(tokens) for tokens in text_indexed]
        text_indexed = [vocab(document.split()) for document in text]
        max_seq = max_seq if max_seq else max(text_lengths)

        self.text_tensor = torch.full(
            (batch_size, max_seq), self.pad_index, dtype=torch.long
        )

        for i, data in enumerate(text_indexed):
            # clip sentence length to max_seq or less to keep space for <sos>, <eos>

            token_ids = (
                text_indexed if text_lengths[i] <= (max_seq - int(add_sos) - int(eos))) else text_indexed[:(max_seq - int(add_sos) - int(eos)))]
            )

            if add_sos:
                token_ids = [self.sos_index,] + token_ids

            if add_eos:
                token_ids = token_ids + [self.eos_index,]
                
            # recalculate because it might be less than max_seq
            seq_len = input_lengths[i] + int(add_sos) + int(add_eos)
            
            self.text_tensor[i, :ls] = torch.tensor(token_ids)
        
        return self.text_tensor


    def decode(self, token_ids: torch.Tensor, keep_specials=True):
        text = []
        for token_id in token_ids:
            tokens = vocab.lookup_tokens(token_id.tolist())
            if not keep_specials:
                tokens = [token for token in tokens if token not in self.special_tokens]
            
            text.append(" ".join(tokens))
        
        return text