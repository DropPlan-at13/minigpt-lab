import torch

text = "hello"

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

print("Characters:", characters)
print("Encoded text:", encoded_text)
print("PyTorch tensor:", data)
print("Tensor shape:", data.shape)
print("Tensor data type:", data.dtype)