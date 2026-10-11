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

context_size = 3

inputs = []
targets = []

for i in range(context_size, len(data)):
    inputs.append(data[i - context_size:i])
    targets.append(data[i])

x = torch.stack(inputs)
y = torch.stack(targets)

print("Input sequences:")
print(x)

print("\nTarget tokens:")
print(y)

print("\nDecoded training examples:")

for input_sequence, target in zip(x, y):
    input_text = "".join(
        index_to_char[token.item()]
        for token in input_sequence
    )

    target_char = index_to_char[target.item()]

    print(f"Input: {input_text} -> Target: {target_char}")

vocab_size = len(characters)
embedding_size = 8

model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(),
    nn.Linear(context_size * embedding_size, 32),
    nn.ReLU(),
    nn.Linear(32, vocab_size)
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

print("\nFinal predictions:")

for input_sequence, target, prediction in zip(
    x, y, predicted_tokens
):
    input_text = "".join(
        index_to_char[token.item()]
        for token in input_sequence
    )

    target_char = index_to_char[target.item()]
    predicted_char = index_to_char[prediction.item()]

    print(
        f"Input: {input_text} | "
        f"Expected: {target_char} | "
        f"Predicted: {predicted_char}"
    )