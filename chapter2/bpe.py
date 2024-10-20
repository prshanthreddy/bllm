from importlib.metadata import version
import tiktoken

print(version('tiktoken'))

tokenizer=tiktoken.get_encoding('gpt2')

text="The quick brown fox jumps over the lazy dog <|endoftext|> The lazy dog jumps over the quickbrownfox"
tokens=tokenizer.encode(text,allowed_special={"<|endoftext|>"})

print("Tokenized :",tokens)

# <|endoftext|> is tokenized as 50256
# Decode the tokens
decoded_text=tokenizer.decode(tokens)

print("After Decoding:",decoded_text)

## Decodes the unknown token " quickbrownfox" as " quickbrownfox" : Correctly tokenized