import logging
import random
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtext import vocab

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super().__init__()

        self.projection_layer = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    def forward(self, encoder_output, decoder_output, src_mask):
        """
        Args:
            encoder_output (torch.Tensor): (N, L, encoder_hidden_size)
            decoder_output (torch.Tensor): (N, 1, decoder_hidden_size)
            src_mask (torch.Tensor): (N, L)

        Returns:
            attention score paid to one decoder token by all (L) the encoder tokens (torch.Tensor): (N, L, 1)
        """
        # => (N, L, decoder_hidden_size)
        projection = self.projection_layer(encoder_output)

        # (N, L, dec_hid_siz) @ (N, dec_hid_siz, 1) => (N, L, 1)
        score = projection @ decoder_output.transpose(1, 2)

        #         score = torch.masked_fill(score, ~src_mask.unsqueeze(dim=2), float("-inf"))
        # => (N, L, 1)
        score = F.softmax(score, dim=1)

        return score


class Encoder(nn.Module):
    def __init__(
            self,
            src_embedding,
            src_embedding_size,
            hidden_size,
            bidirectional=False,
            num_layers=1,
            dropout=0.0,
    ):
        super().__init__()
        self.embedding = src_embedding
        self.layers = nn.Sequential(
            self.embedding,
            nn.LSTM(
                input_size=src_embedding_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
            ),
        )

    def forward(self, src: torch.Tensor):
        """
        Args:
            src (torch.Tensor): (N, Ls) A batch of source sentences represented as tensors.
        """
        # encoder_out (N, L, d), hT => cT => (#direction * #layer, N, d)
        # hidden states from the last timestep
        encoder_out, (last_hidden_state, last_cell_state) = self.layers(src)
        return encoder_out, last_hidden_state


class Decoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size,
            tgt_embedding,
            tgt_embedding_size,
            hidden_size,
            encoder_bidirectional=False,
            num_layers=1,
            dropout=0.0,
    ):
        super().__init__()

        self.embedding = tgt_embedding
        self.lstm = nn.LSTM(
            input_size=tgt_embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = Attention(
            hidden_size * 2 if encoder_bidirectional else hidden_size, hidden_size
        )

        self.decoder_linear = nn.Sequential(
            # 3*hidden_dim because decoder_out and source context will be concatenated
            # this layer is Eq 5 in the Luong et. al. paper
            nn.Linear(
                hidden_size * 3 if encoder_bidirectional else hidden_size, hidden_size
            ),
            nn.Tanh(),
            nn.Linear(hidden_size, tgt_vocab_size),
        )

    def forward(
            self, encoder_out, target, last_hidden_state, last_cell_state, source_mask
    ):
        x = self.embedding(target)
        # => N, 1, d
        output, (ht, ct) = self.lstm(x, (last_hidden_state, last_cell_state))
        # => (N, Ls, 1)
        score = self.attention(encoder_out, output, source_mask)
        # (N, Ls, 1) x (N, Ls, DH) => (N, Ls, DH) => (N, 1, DH)
        # Eq 4 from the Du et al. paper (learning to ask)
        attn_based_ctx = (score * encoder_out).sum(dim=1).unsqueeze(dim=1)
        # => (N, 1, d ) & (N, 1, DH)
        concatenated = torch.cat((output, attn_based_ctx), dim=2).squeeze()
        # => (N, vocab_size)
        logit = self.decoder_linear(concatenated)

        return logit, (ht, ct), score


class Seq2Seq(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            src_embedding_vector,
            tgt_embedding_vector,
            tgt_pad_index,
            tgt_sos_index,
            tgt_eos_index,
            hidden_size,
            bidirectional=True,
            num_layers=2,
            src_embedding_size=100,
            tgt_embedding_size=100,
            dropout=0.3,
    ):

        super().__init__()
        self.tgt_sos_index = tgt_sos_index
        self.tgt_pad_index = tgt_pad_index
        self.tgt_eos_index = tgt_eos_index
        self.tgt_vocab_size = tgt_vocab_size

        self.num_layers = num_layers

        if src_embedding_vector is None:
            self.src_embedding = nn.Embedding(src_vocab_size, src_embedding_size)
        else:
            self.src_embedding = nn.Embedding.from_pretrained(src_embedding_vector)

        if tgt_embedding_vector is None:
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, tgt_embedding_size)
        else:
            self.tgt_embedding = nn.Embedding.from_pretrained(tgt_embedding_vector)

        self.encoder = Encoder(
            self.src_embedding,
            src_embedding_size,
            hidden_size,
            bidirectional,
            num_layers,
            dropout,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            self.tgt_embedding,
            tgt_embedding_size,
            hidden_size,
            bidirectional,
            num_layers,
            dropout,
        )

    def forward(self, source, target, source_mask):
        encoder_out, h = self.encoder(source)

        h = h[: self.num_layers]
        c = torch.randn_like(h, device=device)

        max_seq = target.size(1)
        decoder_input = torch.full(
            (source.size(0), 1), self.tgt_sos_index, device=device, dtype=torch.long
        )
        logits = []

        for t in range(max_seq):
            # (N, tgt_vocab_size)
            logit, (h, c), _ = self.decoder(
                encoder_out, decoder_input, h, c, source_mask
            )
            # (N, 1)
            decoder_input = target[:, t].view(-1, 1)
            logits.append(logit)

        # (N, max_seq, tgt_vocab_size)
        return torch.stack(logits, dim=1)

    def nucleus_generate(self, sources: torch.Tensor, source_masks: torch.Tensor, max_seq: int, p: int):
        """

        :param sources: source tokens (bs, vocab_size)
        :param source_masks: (bs, vocab_size)
        :param max_seq:
        :return:
        """
        N = sources.size(0)

        with torch.no_grad():
            decoder_input = torch.full(
                (N, 1), self.tgt_sos_index, device=device, dtype=torch.long
            )

            # attention scores
            scores: List[torch.Tensor] = []
            # output tokens
            output_token_ids: List[List[int]] = [[self.tgt_sos_index, ] for _ in range(N)]

            # (N, L, DH)
            encoder_out, h = self.encoder(sources)
            h = h[: self.num_layers]
            c = torch.randn_like(h, device=device)
            for t in range(max_seq):
                # (N, tgt_vocab_size)
                # print(encoder_out.shape, decoder_input.shape, h.shape, c.shape)
                logit, (h, c), score = self.decoder(
                    encoder_out, decoder_input, h, c, source_masks
                )

                # (N, 1)
                token_ids, token_probs = nucleus_sample(logit, p)

                for i, token_id in enumerate(token_ids):
                    output_token_ids[i].append(token_id.item())

                scores.append(score)

            return torch.tensor(output_token_ids, dtype=torch.long), torch.stack(scores, dim=2)

    def greedy_generate(self, source, source_mask, max_seq, stop_at_eos=True):
        assert source.size(0) == 1, "requires one sample only"
        with torch.no_grad():
            decoder_input = torch.full(
                (1, 1), self.tgt_sos_index, device=device, dtype=torch.long
            )
            scores = []
            token_ids = []

            # (1, L, DH)
            encoder_out, h = self.encoder(source)
            h = h[: self.num_layers]
            c = torch.randn_like(h, device=device)
            for t in range(max_seq):
                # (1, tgt_vocab_size)
                logit, (h, c), score = self.decoder(
                    encoder_out, decoder_input, h, c, source_mask
                )
                # (1, 1)
                decoder_input = logit.squeeze().argmax(dim=0).view(-1, 1)
                token_ids.append(decoder_input.item())
                scores.append(score)

                if decoder_input.item() == self.tgt_eos_index:
                    break

            return torch.tensor(token_ids, dtype=torch.long), torch.stack(scores, dim=2)

    def beam_generate(self, source, source_mask, k, max_seq, stop_at_eos=True):
        assert source.size(0) == 1, "currently only supports single sample"

        with torch.no_grad():
            # initialize state trackers
            probs = torch.ones((k,), device=device)
            prefix = [[self.tgt_sos_index for _ in range(k)]]

            encoder_out, h = self.encoder(source)
            h = h[: self.num_layers]
            c = torch.randn_like(h, device=device)

            decoder_input = torch.full(
                (1, 1), self.tgt_sos_index, device=device, dtype=torch.long
            )

            # generate first candidates at t=0
            logit, (h, c), _ = self.decoder(
                encoder_out, decoder_input, h, c, source_mask
            )
            logit = logit.detach()
            # with <SOS> as input and k=2, we have top next tokens A, C
            # and their probability P(A), P(C) but they are still not valid
            # prefix as we don't know which one of them will lead to top k
            # probability in the next timestep
            candidate_probs, candidate_prefix = torch.topk(logit, k=k)

            # till now, we have hidden state for one <SOS> token
            # duplicate that for the next k candidates
            h = h.tile(dims=(1, k, 1))
            c = c.tile(dims=(1, k, 1))

            for t in range(1, max_seq):
                # decoder inputs are A, C
                decoder_input = candidate_prefix.view(-1, 1)
                # logits => (k, vocab)
                logit, (h, c), _ = self.decoder(encoder_out, decoder_input, h, c, source_mask)
                # turn into probabilities
                pred_probs = F.softmax(logit, dim=1)
                # multiply A's probability with the predictions where it was outcome
                # same of B as well
                pred_probs = pred_probs.detach() * candidate_probs.unsqueeze(dim=1)

                # to select top k among all the predictions, flatten the prediction
                # probabilities, the shape will be (k*vocab_size, 1)
                pred_probs = pred_probs.flatten()
                k_probs, k_ids = torch.topk(pred_probs, k=k)
                # when flattened,
                # k_ids % vocab_size will give the candidate token index and
                # k_ids // vocab_size will give the prefix index
                prefix_idx = k_ids // self.tgt_vocab_size
                k_token_ids = k_ids % self.tgt_vocab_size

                # store the prefix that produced one of the top k next token
                # if one prefix led to multiple next tokens, it will be repeated
                prefix.append(candidate_prefix[prefix_idx].tolist())

                probs = candidate_probs[prefix_idx] * probs

                # these are the new beam members
                candidate_prefix = k_token_ids
                candidate_probs = k_probs

                # stop at eos
                if stop_at_eos:
                    for i in range(k):
                        if k_token_ids[i] == self.tgt_eos_index:
                            result = [prefix[j][i] for j in range(len(prefix))] + [
                                self.tgt_eos_index,
                            ]
                            prob = probs[i] * k_probs[i]
                            return torch.tensor(result).view(1, -1), prob

            # reorganize the output prefix
            i = probs.argmax()
            result = [prefix[j][i] for j in range(len(prefix))]

            return torch.tensor(result).view(1, -1), probs[i]



def nucleus_sample(logits: torch.Tensor, p: float) -> (List[int], List[float]):
    """

    :rtype: Tuple
    """
    assert logits.dim() == 2, "expected a matrix (batch, vocab_size)"

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_sum = torch.cumsum(sorted_probs, dim=-1)
    out_of_nucleus = cumulative_sum > p
    # cumulative_sum > p comparison always misses the last token that should be in the nucleus
    # this line fixes that
    out_of_nucleus[:, 1:] = out_of_nucleus[:, :-1].clone()
    out_of_nucleus[:, 0] = False
    sorted_probs[out_of_nucleus] = 0
    # Eq. 3 from the nucleus paper
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1).unsqueeze(1)
    sorted_selected_indices = torch.multinomial(sorted_probs, 1)
    token_probs = torch.gather(sorted_probs, dim=-1, index=sorted_selected_indices)
    token_indices = torch.gather(sorted_indices, dim=-1, index=sorted_selected_indices)

    # (N, 1), (N, 1)
    return token_indices.detach(), token_probs.detach()


def generate(model, sentence, src_tokenizer, tgt_tokenizer, cfg, method="greedy", p=0.8) -> (
str, Optional[torch.Tensor]):
    """
    Given a model and source text, generate the target text.

    Returns:
        text: str
        attention_score: when method is greedy
.flatten().tolist()
    """
    if method in ["greedy", "beam"]:
        sentence = [sentence]

    src_token_ids, src_mask = src_tokenizer.encode(
        sentence, max_seq=cfg["src_max_seq"]
    )
    src_token_ids = src_token_ids.to(device)
    src_mask = src_mask.to(device)

    attention_scores = None
    if method == "greedy":
        tgt_token_ids, attention_scores = model.greedy_generate(
            src_token_ids, src_mask, cfg["tgt_max_seq"], stop_at_eos=True
        )
        attention_scores = attention_scores.squeeze(dim=3)
        tokens = tgt_tokenizer.decode(tgt_token_ids.view(1, -1), keep_specials=False)
        return tokens, attention_scores

    elif method == "beam":
        tgt_token_ids, prob = model.beam_generate(
            src_token_ids, src_mask, cfg['beam_size'], cfg["tgt_max_seq"], stop_at_eos=True
        )
        tokens = tgt_tokenizer.decode(tgt_token_ids.view(1, -1), keep_specials=False)
        return tokens, attention_scores

    elif method == "nucleus":
        tgt_token_ids, attention_scores = model.nucleus_generate(
            src_token_ids, src_mask, cfg["tgt_max_seq"], p
        )
        return tgt_tokenizer.decode(tgt_token_ids, keep_specials=False), attention_scores


def init_model(cfg, src_vocab, tgt_vocab):
    src_embedding_vector = None
    tgt_embedding_vector = None

    if cfg["glove_dir"] is not None:
        glove = vocab.Vectors(name='glove.840B.300d.txt', cache=cfg['glove_dir'])
        src_embedding_vector = glove.get_vecs_by_tokens(src_vocab.get_itos())
        tgt_embedding_vector = glove.get_vecs_by_tokens(tgt_vocab.get_itos())

    model = Seq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_embedding_vector=src_embedding_vector,
        tgt_embedding_vector=tgt_embedding_vector,
        tgt_pad_index=tgt_vocab["<PAD>"],
        tgt_sos_index=tgt_vocab["<SOS>"],
        tgt_eos_index=tgt_vocab["<EOS>"],
        hidden_size=cfg["hidden_size"],
        bidirectional=cfg["bidirectional"],
        num_layers=cfg["num_layers"],
        src_embedding_size=cfg["src_embedding_size"],
        tgt_embedding_size=cfg["tgt_embedding_size"],
        dropout=cfg["dropout"], )

    model.to(device)
    return model
