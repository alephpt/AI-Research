import os

# open tokens.txt and deduplicate any tokens
with open("tokens.txt", "r", encoding="utf-8") as f:
    tokens = f.readlines()

print("Tokens Start: " + str(len(tokens)))
tokens = list(set(tokens))
tokens.sort(key=len, reverse=True)

print("Tokens Finish: " + str(len(tokens)))

# save to dedup_tokens.txt
with open("dedup_tokens.txt", "w") as f:
    f.writelines(tokens)
