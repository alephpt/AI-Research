import os

# open tokens.txt and deduplicate any tokens
with open("tokens.txt", "r") as f:
    tokens = f.readlines()

tokens = sorted(list(set(tokens)))

print("Tokens: " + str(len(tokens)))

# save to dedup_tokens.txt
with open("dedup_tokens.txt", "w") as f:
    f.writelines(tokens)
