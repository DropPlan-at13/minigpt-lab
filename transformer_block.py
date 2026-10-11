import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, context_size):
        super().__init__()

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_size, context_size))
        )

        self.embedding_size = embedding_size

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.embedding_size)

        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        output = attention_weights @ values
        output = self.output_projection(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, context_size):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        self.attention = CausalSelfAttention(
            embedding_size,
            context_size
        )

        self.feed_forward = FeedForward(embedding_size)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


torch.manual_seed(42)

batch_size = 2
sequence_length = 4
embedding_size = 8
context_size = 16

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_size
)

block = TransformerBlock(
    embedding_size,
    context_size
)

output = block(x)

print("Input shape:", x.shape)
print("Transformer output shape:", output.shape)

print("\nFirst sequence output:")
print(output[0])

print("\nTotal trainable parameters:")
print(
    sum(
        parameter.numel()
        for parameter in block.parameters()
        if parameter.requires_grad
    )
)