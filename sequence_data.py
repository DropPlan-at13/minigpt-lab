import torch

text = "hello"

characters = sorted(set(text))

char_to_index = {
    char: index
    for index, char in enumerate(characters)
}

encoded_text = [
    char_to_index[char]
    for char in text
]

data = torch.tensor(encoded_text, dtype=torch.long)

x = data[:-1]
y = data[1:]

print("Original data:", data)
print("Input tensor:", x)
print("Target tensor:", y)

print("Input shape:", x.shape)
print("Target shape:", y.shape)