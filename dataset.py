import torch

with open("/home/kishore/minigpt-lab/data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

characters = sorted(set(text))

char_to_index = {
    char: index
    for index, char in enumerate(characters)
}

index_to_char = {
    index: char
    for char, index in char_to_index.items()
}

encoded_text = [
    char_to_index[char]
    for char in text
]

data = torch.tensor(encoded_text, dtype=torch.long)

context_size = 16

x = []
y = []

for i in range(len(data) - context_size):
    x.append(data[i:i + context_size])
    y.append(data[i + context_size])

x = torch.stack(x)
y = torch.stack(y)

print("Total characters:", len(text))
print("Vocabulary size:", len(characters))
print("Encoded data shape:", data.shape)
print("Input dataset shape:", x.shape)
print("Target dataset shape:", y.shape)

print("\nFirst training example:")

first_input = "".join(
    index_to_char[token.item()]
    for token in x[0]
)

first_target = index_to_char[y[0].item()]

print("Input:", repr(first_input))
print("Target:", repr(first_target))