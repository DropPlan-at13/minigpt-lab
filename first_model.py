import torch
import torch.nn as nn

torch.manual_seed(42)

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

x = data[:-1]
y = data[1:]

vocab_size = len(characters)
embedding_size = 8

model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    nn.Linear(embedding_size, vocab_size)
)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 500

for epoch in range(epochs):
    predictions = model(x)

    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )

print("\nTraining complete!")

with torch.no_grad():
    predictions = model(x)
    predicted_tokens = predictions.argmax(dim=1)

print("\nInput characters:")

for token in x:
    print(index_to_char[token.item()], end=" ")

print("\n\nCorrect next characters:")

for token in y:
    print(index_to_char[token.item()], end=" ")

print("\n\nPredicted next characters:")

for token in predicted_tokens:
    print(index_to_char[token.item()], end=" ")

print()