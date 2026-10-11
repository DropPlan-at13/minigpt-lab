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
        _, sequence_length, _ = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.embedding_size)

        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        output = attention_weights @ values

        return self.output_projection(output)


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


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=32,
        context_size=32
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_size
        )

        self.transformer = TransformerBlock(
            embedding_size,
            context_size
        )

        self.final_norm = nn.LayerNorm(embedding_size)

        self.language_head = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_size:
            raise ValueError(
                "Input sequence exceeds the context size."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        x = self.transformer(x)
        x = self.final_norm(x)

        logits = self.language_head(x)

        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100):
        self.eval()

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.context_size:]

            logits = self(context)

            next_token_logits = logits[:, -1, :]

            probabilities = F.softmax(
                next_token_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            token_ids = torch.cat(
                (token_ids, next_token),
                dim=1
            )

        return token_ids