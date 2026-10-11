import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import CharacterTokenizer
from minigpt import MiniGPT


torch.manual_seed(42)

device = torch.device("cpu")

with open("data/input.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer = CharacterTokenizer(text)

encoded_text = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

split_index = int(len(encoded_text) * 0.8)

train_data = encoded_text[:split_index]
validation_data = encoded_text[split_index:]

context_size = 32
embedding_size = 32
batch_size = 16
epochs = 300
learning_rate = 0.001


def create_sequences(data, context_size):
    inputs = []
    targets = []

    for i in range(len(data) - context_size):
        inputs.append(data[i:i + context_size])
        targets.append(data[i + 1:i + context_size + 1])

    if not inputs:
        raise ValueError(
            "Not enough characters to create a sequence."
        )

    return torch.stack(inputs), torch.stack(targets)


x_train, y_train = create_sequences(
    train_data,
    context_size
)

x_validation, y_validation = create_sequences(
    validation_data,
    context_size
)

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    embedding_size=embedding_size,
    context_size=context_size
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


def calculate_loss(inputs, targets):
    logits = model(inputs)

    loss = F.cross_entropy(
        logits.reshape(-1, tokenizer.vocab_size),
        targets.reshape(-1)
    )

    return loss


print("Device:", device)
print("Total characters:", len(encoded_text))
print("Vocabulary size:", tokenizer.vocab_size)
print("Training characters:", len(train_data))
print("Validation characters:", len(validation_data))
print("Training sequences:", len(x_train))
print("Validation sequences:", len(x_validation))
print(
    "Trainable parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
)

for epoch in range(epochs):
    model.train()

    permutation = torch.randperm(len(x_train))
    total_training_loss = 0.0

    for start in range(0, len(x_train), batch_size):
        indices = permutation[start:start + batch_size]

        x_batch = x_train[indices]
        y_batch = y_train[indices]

        optimizer.zero_grad()

        loss = calculate_loss(x_batch, y_batch)

        loss.backward()
        optimizer.step()

        total_training_loss += loss.item() * len(indices)

    average_training_loss = (
        total_training_loss / len(x_train)
    )

    model.eval()

    with torch.no_grad():
        validation_loss = calculate_loss(
            x_validation,
            y_validation
        ).item()

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train loss: {average_training_loss:.4f} "
            f"| Validation loss: {validation_loss:.4f}"
        )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "embedding_size": embedding_size,
        "context_size": context_size,
        "vocabulary": tokenizer.vocabulary,
    },
    "minigpt_model.pth"
)

print("\nModel saved to minigpt_model.pth")