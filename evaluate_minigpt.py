import math
import torch
import torch.nn.functional as F

from tokenizer import CharacterTokenizer
from minigpt import MiniGPT


device = torch.device("cpu")

with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = CharacterTokenizer(text)

checkpoint = torch.load(
    "minigpt_model.pth",
    map_location=device,
    weights_only=True
)

model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    embedding_size=checkpoint["embedding_size"],
    context_size=checkpoint["context_size"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

context_size = checkpoint["context_size"]

encoded_text = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
    device=device
)

inputs = encoded_text[:-1].unfold(
    0,
    context_size,
    1
)

targets = encoded_text[1:].unfold(
    0,
    context_size,
    1
)

with torch.no_grad():
    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, tokenizer.vocab_size),
        targets.reshape(-1)
    )

    perplexity = math.exp(loss.item())

print("Evaluation sequences:", inputs.size(0))
print("Predictions:", logits.numel() // tokenizer.vocab_size)
print(f"Cross-entropy loss: {loss.item():.4f}")
print(f"Perplexity: {perplexity:.4f}")