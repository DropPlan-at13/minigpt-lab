import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embedding = nn.Embedding(
            context_size,
            embedding_size
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        token_vectors = self.token_embedding(token_ids)

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_vectors = self.position_embedding(positions)

        embeddings = token_vectors + position_vectors

        return embeddings


torch.manual_seed(42)

vocab_size = 25
embedding_size = 8
context_size = 16

token_ids = torch.tensor([
    [17, 14, 1, 14],
    [12, 4, 17, 13]
])

embedding_layer = TokenAndPositionEmbedding(
    vocab_size,
    embedding_size,
    context_size
)

embeddings = embedding_layer(token_ids)

print("Token IDs shape:", token_ids.shape)
print("Embedding output shape:", embeddings.shape)

print("\nFirst token embedding:")
print(embeddings[0, 0])

print("\nFirst sequence embeddings:")
print(embeddings[0])