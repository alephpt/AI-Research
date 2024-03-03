import re, os

# open max_tokens.txt
with open("max_tokens.txt", "r", encoding="utf-8") as f:
    tokens = f.read().splitlines()
    
print("Tokens: " + str(len(tokens)))
    
strings = []
# open all the text files in ../training_data
for file in os.listdir("../training_data"):
    with open("../training_data/" + file, "r", encoding="utf8") as f:
        strings += f.read().splitlines()

print("lines in strings: " + str(len(strings)))    

unique_words = []
for line in strings:
    words = line.split(' ')
    
    for word in words:
        word = re.sub(r'[^a-zA-Z]', '', word)
        unique_words.append(word)
 
unique_words = set(unique_words)
print("Unique Words: " + str(len(unique_words)))
 
for token in tokens:
    if len(token) == 1:
        continue
    
    token_found = False
    
    for word in unique_words:
        if len(token) > len(word):
            continue
        
        if token in word:
            print("Found \"" + token + "\" in \"" + word + "\"")
            token_found = True
            break
        
    if not token_found:
        print("Removing \"" + token + "\"")
        tokens.remove(token)
        
print("Tokens: " + str(len(tokens)))

with open("max_cleaned_tokens.txt", "w", encoding="utf-8") as f:
    f.write(str('\n'.join(tokens)))