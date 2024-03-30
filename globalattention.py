import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super().__init__()

        self.projection_layer = nn.Linear(encoder_hidden_size, decoder_hidden_size)

    def forward(self, encoder_output, decoder_output):
        """
        Args:
            encoder_output (torch.Tensor): (N, L, encoder_hidden_size)
            decoder_output (torch.Tensor): (N, 1, decoder_hidden_size)


        Returns:
            score (torch.Tensor): (N, L, 1)
        """
        # => (N, L, decoder_hidden_size)
        projection = self.projection_layer(encoder_output)
        # => (N, L, 1)
        score = projection @ decoder_output.transpose(1, 2)
        score = F.softmax(score, dim=1)

        return score
