import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1)

        scores = scores / math.sqrt(embedding_size)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        output = attention_weights @ values

        output = self.output_projection(output)

        return output, attention_weights


torch.manual_seed(42)

embedding_size = 8
batch_size = 1
sequence_length = 4

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_size
)

attention = CausalSelfAttention(embedding_size)

output, weights = attention(x)

print("Input shape:", x.shape)
print("Attention output shape:", output.shape)

print("\nAttention weights:")
print(weights)

print("\nOutput:")
print(output)