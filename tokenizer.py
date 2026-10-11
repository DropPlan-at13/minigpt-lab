class CharacterTokenizer:
    def __init__(self, text):
        self.vocabulary = sorted(set(text))

        self.char_to_index = {
            char: index
            for index, char in enumerate(self.vocabulary)
        }

        self.index_to_char = {
            index: char
            for char, index in self.char_to_index.items()
        }

        self.vocab_size = len(self.vocabulary)

    def encode(self, text):
        return [
            self.char_to_index[char]
            for char in text
        ]

    def decode(self, tokens):
        return "".join(
            self.index_to_char[token]
            for token in tokens
        )