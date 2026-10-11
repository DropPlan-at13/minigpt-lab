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

attention_weights = F.softmax(scores, dim=-1)

output = attention_weights @ values

print("Input token vectors:")
print(tokens)

print("\nAttention scores:")
print(scores)

print("\nAttention weights:")
print(attention_weights)

print("\nAttention output:")
print(output)