import tiktoken
tokenizer=tiktoken.get_encoding('gpt2')

with open("/Users/prashanthreddy/Documents/GitHub/bllm/chapter2/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enctext=tokenizer.encode(raw_text)
enctext=enctext[50:]
# print("Encoded text:",enctext[50:])


# dec_text=tokenizer.decode(enctext)
# print("Decoded text:",dec_text[:50])


context_size=4
x=enctext[:context_size]
y=enctext[1:context_size+1]
# print("x:",x)
# print("y:",y)
print("-----------")    
for i in range(1,context_size+1):
    x=enctext[:i]
    y=enctext[i]
    # print("x:",x)
    # print("y:",y)
    # print(x,"----->",y)
    # COnverting into text
    xtext=tokenizer.decode(x)
    ytext=tokenizer.decode([y])
    print(xtext,"----->",ytext)
