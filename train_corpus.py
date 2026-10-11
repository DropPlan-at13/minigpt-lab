import torch
import torch.nn as nn

torch.manual_seed(42)

with open("data/input.txt", "r", encoding="utf-8") as file:
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
embedding_size = 16
hidden_size = 64
batch_size = 32

x = []
y = []

for i in range(len(data) - context_size):
    x.append(data[i:i + context_size])
    y.append(data[i + context_size])

x = torch.stack(x)
y = torch.stack(y)

vocab_size = len(characters)

model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(),
    nn.Linear(context_size * embedding_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, vocab_size)
)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.003
)

epochs = 300

for epoch in range(epochs):
    permutation = torch.randperm(len(x))

    for start in range(0, len(x), batch_size):
        indices = permutation[start:start + batch_size]

        batch_x = x[indices]
        batch_y = y[indices]

        predictions = model(batch_x)

        loss = loss_function(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {loss.item():.4f}"
        )

print("\nTraining complete!")

model.eval()

with torch.no_grad():
    sample = x[0:1]
    prediction = model(sample)
    predicted_token = prediction.argmax(dim=1).item()

sample_text = "".join(
    index_to_char[token.item()]
    for token in sample[0]
)

print("\nSample prediction:")
print("Input:", repr(sample_text))
print("Expected:", repr(index_to_char[y[0].item()]))
print("Predicted:", repr(index_to_char[predicted_token]))