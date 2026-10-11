text = "hello"

for i in range(1, len(text)):
    input_text = text[:i]
    target_char = text[i]

    print("Input:", input_text)
    print("Target:", target_char)
    print()