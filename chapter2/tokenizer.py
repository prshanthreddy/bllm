with open("chapter2/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])


import re
def tokenize(text):
    return re.findall(r"\b\w+\b", text)
tokens = tokenize(raw_text)
print("Total number of tokens:", len(tokens))
print(tokens[:10])