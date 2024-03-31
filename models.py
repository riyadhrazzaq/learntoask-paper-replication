import torch
import torchtext
from torch import nn
import torch.nn.functional as F

from globalattention import Attention

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
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
            ),
        )

    def forward(self, src: torch.Tensor):
        """
        Args:
            src (torch.Tensor): (N, Ls) A batch of source sentences represented as tensors.
        """
        # encoder_representation (N, Ls, d), hT => cT => (#direction * #layer, N, d) : hidden states from the last timestep
        encoder_out, (last_hidden_state, last_cell_state) = self.encoder(src)
        return encoder_out, (last_hidden_state, last_cell_state)


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
        )

        # self.attn_layer = nn.Linear(
        #     in_features=hidden_dim * 2 if bidirectional else hidden_dim,
        #     out_features=hidden_dim * 2 if bidirectional else hidden_dim,
        # )

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

        return logit, (ht, ct), score


class Seq2SeqEncoderDecoder(nn.Module):
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

        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(embedding_vector)
        self.encoder = Encoder(
            vocab_size,
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
        encoder_out, (h, c) = self.encoder(source)
        h = h[: self.num_layers]
        c = c[: self.num_layers]
        max_seq = target.size(1)
        logits = []
        scores = []

        for timestep in range(max_seq):
            logit, (h, c), score = self.decoder(
                encoder_out, target[:, timestep].unsqueeze(dim=1), h, c
            )
            logits.append(logit)
            scores.append(score)

        return logits, scores

    def generate(self, source, max_seq=10):
        encoder_out, (h, c) = self.encoder(source)
        h = h[: self.num_layers]
        c = c[: self.num_layers]

        generated_seq = [
            torch.tensor(
                [self.sos_index for _ in range(source.size(0))],
                dtype=torch.long,
                device=device,
            )
        ]

        scores = []
        for timestep in range(max_seq):
            target = generated_seq[-1].view(-1, 1)
            logit, (h, c), score = self.decoder(encoder_out, target, h, c)
            scores.append(score)
            most_probable_tokens = torch.max(logit, dim=1)[1]
            generated_seq.append(most_probable_tokens)

        return torch.concat(generated_seq, dim=0).view(source.size(0), -1), scores
