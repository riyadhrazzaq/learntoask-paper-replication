import torch
import torchtext
from torch import nn
import torch.nn.functional as F

from attention import Attention

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(
        self,
        embedding,
        embedding_dim,
        hidden_dim=8,
        bidirectional=False,
        num_layers=1,
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = nn.Sequential(
            self.embedding,
            nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=0.3 if num_layers > 1 else 0.0,
            ),
        )

    def forward(self, src: torch.Tensor):
        """
        Args:
            src (torch.Tensor): (N, Ls) A batch of source sentences represented as tensors.
        """
        # encoder_representation (N, Ls, d), hT => cT => (#direction * #layer, N, d) : hidden states from the last timestep
        encoder_out, (last_hidden_state, last_cell_state) = self.encoder(src)
        return encoder_out, last_hidden_state


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding,
        embedding_dim,
        hidden_dim=8,
        encoder_bidirectional=False,
        num_layers=1,
    ):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=0.3 if num_layers > 1 else 0.0,
        )

        self.attention = Attention(
            hidden_dim * 2 if encoder_bidirectional else hidden_dim, hidden_dim
        )

        self.decoder_linear = nn.Sequential(
            # 3*hidden_dim because decoder_out and source context will be concatenated
            # this layer is Eq 5 in the Luong et. al. paper
            nn.Linear(
                hidden_dim * 3 if encoder_bidirectional else hidden_dim, hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, encoder_out, target, last_hidden_state, last_cell_state):
        x = self.embedding(target)
        # => N, 1, d
        output, (ht, ct) = self.lstm(x, (last_hidden_state, last_cell_state))
        # => (N, Ls, 1)
        score = self.attention(encoder_out, output)
        # (N, Ls, 1) x (N, Ls, DH) => (N, Ls, DH) => (N, 1, DH)
        # Eq 4 from the Du et. al. paper (learning to ask)
        attn_based_ctx = (score * encoder_out).sum(dim=1).unsqueeze(dim=1)
        # => (N, 1, d ) & (N, 1, DH)
        concatenated = torch.cat((output, attn_based_ctx), dim=2).squeeze()
        # => (N, vocab_size)
        logit = self.decoder_linear(concatenated)

        return logit, (ht, ct)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_vector,
        embedding_dim,
        pad_index,
        sos_index,
        eos_index,
        hidden_dim=8,
        bidirectional=True,
        num_layers=2,
    ):

        super().__init__()
        self.sos_index = sos_index
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.vocab_size = vocab_size

        self.num_layers = num_layers
        if embedding_vector is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding_vector)
        self.encoder = Encoder(
            self.embedding,
            embedding_dim,
            hidden_dim,
            bidirectional,
            num_layers,
        )
        self.decoder = Decoder(
            vocab_size,
            self.embedding,
            embedding_dim,
            hidden_dim,
            bidirectional,
            num_layers,
        )

    def forward(self, source, target):
        encoder_out, h = self.encoder(source)
        h = h[: self.num_layers]
        c = torch.randn_like(h, device=device)
        max_seq = target.size(1)
        decoder_input = torch.full(
            (source.size(0), 1), self.sos_index, device=device, dtype=torch.long
        )
        logits = []

        for t in range(max_seq):
            logit, (h, c) = self.decoder(encoder_out, decoder_input, h, c)
            decoder_input = target[:, t].view(-1, 1)
            logits.append(logit)

        return torch.stack(logits, dim=1)

    def generate_batch(self, source, method="greedy", max_seq=15):
        encoder_out, h = self.encoder(source)
        h = h[: self.num_layers]
        c = torch.randn_like(h, device=device)

        decoder_input = torch.full(
            (source.size(0), 1), self.sos_index, device=device, dtype=torch.long
        )
        logits = []
        outputs = torch.full(
            (source.size(0), max_seq), self.pad_index, device=device, dtype=torch.long
        )

        for t in range(max_seq):
            logit, (h, c) = self.decoder(encoder_out, decoder_input, h, c)
            print(logit.shape)
            most_probable_tokens = torch.max(logit, dim=1)[1]
            outputs[:, t] = most_probable_tokens
            decoder_input = most_probable_tokens.view(-1, 1)

        return outputs

    def beam_generate(self, source, k=3, max_seq=15, stop_at_eos=True):
        assert source.size(0) == 1, "currently only supports single sample"
        # initialize state trackers
        probs = torch.ones((k,))
        prefix = [[self.sos_index for _ in range(k)]]

        encoder_out, h = self.encoder(source)
        h = h[: self.num_layers]
        c = torch.randn_like(h, device=device)

        decoder_input = torch.full(
            (source.size(0), 1), self.sos_index, device=device, dtype=torch.long
        )

        # generate first candidates at t=0
        logit, (h, c) = self.decoder(encoder_out, decoder_input, h, c)
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
            logit, (h, c) = self.decoder(encoder_out, decoder_input, h, c)
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
            prefix_idx = k_ids // self.vocab_size
            k_token_ids = k_ids % self.vocab_size

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
                    if k_token_ids[i] == self.eos_index:
                        result = [prefix[j][i] for j in range(len(prefix))] + [
                            self.eos_index,
                        ]
                        prob = probs[i] * k_probs[i]
                        return result, prob

        # reorganize the output prefix
        i = probs.argmax()
        result = [prefix[j][i] for j in range(len(prefix))]

        return torch.tensor(result).view(1, -1), probs[i]
