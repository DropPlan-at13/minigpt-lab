import torch
import torch.nn.functional as F

torch.manual_seed(42)

tokens = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

queries = tokens
keys = tokens
values = tokens

scores = queries @ keys.T

sequence_length = tokens.size(0)

mask = torch.tril(
    torch.ones(sequence_length, sequence_length)
)

masked_scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

attention_weights = F.softmax(
    masked_scores,
    dim=-1
)

output = attention_weights @ values

print("Causal mask:")
print(mask)

print("\nMasked attention scores:")
print(masked_scores)

print("\nCausal attention weights:")
print(attention_weights)

print("\nCausal attention output:")
print(output)