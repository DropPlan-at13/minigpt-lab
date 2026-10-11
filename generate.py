import torch

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

prompt = "artificial"

unknown_characters = [
    char for char in prompt
    if char not in tokenizer.char_to_index
]

if unknown_characters:
    raise ValueError(
        f"Prompt contains characters outside the vocabulary: "
        f"{unknown_characters}"
    )

input_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)

torch.manual_seed(42)

generated_ids = model.generate(
    input_ids,
    max_new_tokens=100
)

generated_text = tokenizer.decode(
    generated_ids[0].tolist()
)

print("Prompt:", repr(prompt))
print("\nGenerated text:")
print(generated_text)